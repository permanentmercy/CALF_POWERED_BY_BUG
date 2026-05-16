import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import os
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model
from models.GPT2_arch import AccustumGPT2Model
def load_gpt2_model(model_class, model_path=None, **kwargs):
    """优先从指定路径或本地缓存加载GPT2模型，如果没有才从huggingface下载

    Args:
        model_class: 模型类（例如 `AccustumGPT2Model`）
        model_path: 可选，模型路径或模型名（本地目录或huggingface id）
    """
    # 如果用户显式指定了路径或id，优先使用
    if model_path:
        try:
            print(f"trying to load GPT2 model from provided path/id: {model_path} (local only)...")
            return model_class.from_pretrained(model_path, local_files_only=True, **kwargs)
        except (OSError, FileNotFoundError, ConnectionError):
            try:
                print(f"local model at {model_path} not found, attempting to download from huggingface...")
                return model_class.from_pretrained(model_path, **kwargs)
            except Exception as e:
                print(f"failed to load model from provided path/id {model_path}: {e}")
                raise

    # 否则保持原有逻辑，先尝试本地缓存的 'gpt2'，再回退到下载
    try:
        print("trying to load GPT2 model from local cache (gpt2)...")
        return model_class.from_pretrained('gpt2', local_files_only=True, **kwargs)
    except (OSError, FileNotFoundError, ConnectionError):
        print("local GPT2 model not found, downloading from huggingface (gpt2)...")
        return model_class.from_pretrained('gpt2', **kwargs)

class Encoder_PCA(nn.Module):
    def __init__(self, input_dim, word_embedding, hidden_dim=768, num_heads=12, num_encoder_layers=1, dim_feedforward=2048, cycle_len=24, use_tq_gate=False, tq_dropout=0.0, tq_mode='mul', use_tq=True):
        super(Encoder_PCA, self).__init__()
        self.use_tq_gate = use_tq_gate
        self.tq_mode = tq_mode
        self.use_tq = use_tq
        self.linear = nn.Linear(input_dim, hidden_dim)

        self.temporal_query = nn.Parameter(torch.randn(cycle_len, hidden_dim))
        self.tq_dropout = nn.Dropout(tq_dropout)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attention': nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads),
                'norm1': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, dim_feedforward),
                    nn.GELU(),
                    nn.Dropout(tq_dropout),
                    nn.Linear(dim_feedforward, hidden_dim),
                ),
                'norm2': nn.LayerNorm(hidden_dim),
                'dropout': nn.Dropout(tq_dropout),
            }) for _ in range(num_encoder_layers)
        ])
        
        # 注册为 buffer，确保其随模型移动到 GPU 且不被视为参数
        self.register_buffer('word_embedding_base', word_embedding.T)

        # 门控参数：控制时间信息的注入强度，初始化为0 (sigmoid后为0.5)
        if self.use_tq_gate:
            self.tq_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, cycle_index=None):
        B = x.shape[0]
        
        # 使用 expand 得到 (B, Prototypes, Hidden) 的视图
        w_embed = self.word_embedding_base.unsqueeze(0).expand(B, -1, -1)

        # 核心开关：只有在 use_tq 为 True 且提供了索引时，才注入周期性向量
        if self.use_tq and cycle_index is not None:
            tq = self.temporal_query[cycle_index] # (B, 768)
            if self.tq_mode == 'mul':
                w_embed = w_embed * (torch.sigmoid(self.tq_dropout(tq.unsqueeze(1))) * 2)
            else:
                w_embed = w_embed + self.tq_dropout(tq.unsqueeze(1))

        x = self.linear(x)
        x_time = x

        # 多层特征提取
        for layer in self.layers:
            # Cross-Attention: Query from x, Key/Value from w_embed
            q = x.transpose(0, 1)
            k = v = w_embed.transpose(0, 1)
            
            attn_output, _ = layer['cross_attention'](q, k, v)
            x = x + layer['dropout'](attn_output.transpose(0, 1))
            x = layer['norm1'](x)
            
            # Feed-Forward Network
            ffn_output = layer['ffn'](x)
            x = x + layer['dropout'](ffn_output)
            x = layer['norm2'](x)

        if self.use_tq_gate:
            # 门控融合：(1-w) * 原始特征 + w * TQ增强特征
            # 设置 0.1 的保底值，强制 TQ 信息参与，防止门控完全坍塌
            gate_weight = 0.1 + 0.9 * torch.sigmoid(self.tq_gate)
            x_fused = (1 - gate_weight) * x_time + gate_weight * x
            return x_time, x_fused
        else:
            # 之前的老逻辑：直接使用 TQ 增强后的特征
            return x_time, x

class TQ_OutputHead(nn.Module):
    def __init__(self, d_model, output_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, x, res_w=0.0):
        # x: (batch, seq, d_model)
        h = self.mlp(x)
        if res_w != 0:
            h = h + res_w * x
        return self.output_proj(h)

class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=["c_attn"]
        )
    
        self.task_name = configs.task_name
        # 支持从外部传入模型路径/ID（优先）
        model_path = getattr(configs, 'gpt2_path', None)
        self.gpt2 = load_gpt2_model(AccustumGPT2Model, model_path=model_path, output_attentions=True, output_hidden_states=True)
        self.gpt2_text = load_gpt2_model(AccustumGPT2Model, model_path=model_path, output_attentions=True, output_hidden_states=True)

        # 物理裁剪层数并释放显存
        # 文本分支：永远从第 0 层开始取
        if len(self.gpt2_text.h) > configs.gpt_layers:
            print(f">>> Pruning GPT-2 Text: keeping layers [0:{configs.gpt_layers}]")
            self.gpt2_text.h = nn.ModuleList([self.gpt2_text.h[i] for i in range(configs.gpt_layers)])
            
        # 时间分支：根据 layer_offset 决定起始层
        layer_offset = getattr(configs, 'layer_offset', 0)
        total_available = len(self.gpt2.h)
        if layer_offset + configs.gpt_layers <= total_available:
            print(f">>> Pruning GPT-2 Time: keeping layers [{layer_offset}:{layer_offset + configs.gpt_layers}]")
            self.gpt2.h = nn.ModuleList([self.gpt2.h[i + layer_offset] for i in range(configs.gpt_layers)])
        else:
            print(f">>> Warning: layer_offset {layer_offset} is too large, falling back to layer 0.")
            self.gpt2.h = nn.ModuleList([self.gpt2.h[i] for i in range(configs.gpt_layers)])
        
        # 强制垃圾回收
        gc.collect()
        torch.cuda.empty_cache()

        self.gpt2 = get_peft_model(self.gpt2, peft_config)#LORA微调参数传入
        
        word_embedding = torch.tensor(torch.load(configs.word_embedding_path)).to(device=device)
        
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name or 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        for i, (name, param) in enumerate(self.gpt2_text.named_parameters()):
            if 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.time_proj = nn.ModuleList([nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers+1)])
        
        self.text_proj = nn.ModuleList([nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers+1)])

        # 把 configs.d_ff 传进 Encoder_PCA，让 --d_ff 生效
        self.in_layer = Encoder_PCA(
            configs.seq_len,                  # 输入维度（线性层输入）
            word_embedding,                  # 用于交叉注意力的词嵌入
            configs.d_model,    # Transformer 的 d_model
            num_encoder_layers=configs.e_layers, # 编码器层数
            dim_feedforward=configs.d_ff,  # Transformer FFN 中间维度
            cycle_len=configs.cycle,         # TQ 机制的循环长度
            use_tq_gate=configs.use_tq_gate, # 是否使用门控
            tq_dropout=getattr(configs, 'tq_dropout', 0.0), # TQ Dropout
            tq_mode=getattr(configs, 'tq_mode', 'mul'),     # TQ 调制模式: add/mul
            use_tq=getattr(configs, 'use_tq', 1)            # 是否开启 TQ 注入
        )
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.out_layer = TQ_OutputHead(configs.d_model, configs.pred_len, configs.dropout)
        elif self.task_name == 'classification':
            self.out_layer = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
            self.mlp_res_proj = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
        elif self.task_name == 'imputation':
            self.out_layer = TQ_OutputHead(configs.d_model, configs.seq_len, configs.dropout)
        elif self.task_name == 'anomaly_detection':
            self.out_layer = TQ_OutputHead(configs.d_model, configs.seq_len, configs.dropout)
        
        self.mlp_res_w = getattr(configs, 'mlp_res_w', 0.0)

        for layer in (self.gpt2_text, self.gpt2, self.in_layer, self.out_layer, self.time_proj, self.text_proj):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0
        

    def forecast(self, x, cycle_index=None):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x, cycle_index=cycle_index)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
        # residue connection
        outputs_time += outputs_time1
        outputs_text += outputs_text1
        
        intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        outputs_time_feat = outputs_time[:, -M:, :]
        outputs_text_feat = outputs_text[:, -M:, :]

        outputs_time = self.out_layer(outputs_time_feat, res_w=self.mlp_res_w)
        outputs_text = self.out_layer(outputs_text_feat, res_w=self.mlp_res_w)

        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time':outputs_time,
            'intermidiate_time':intermidiate_feat_time,
            'intermidiate_text':intermidiate_feat_text,
        }


    def classification(self, x, cycle_index=None):
        B, L, M = x.shape

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x, cycle_index=cycle_index)
        
        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
        
        outputs_time += outputs_time1
        outputs_text += outputs_text1
        
        intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])
        
        outputs_time_res = outputs_time.reshape(B, -1)
        outputs_text_res = outputs_text.reshape(B, -1)
        
        outputs_time = self.out_layer(outputs_time_res)
        outputs_text = self.out_layer(outputs_text_res)
        
        if self.mlp_res_w != 0:
            outputs_time = outputs_time + self.mlp_res_w * self.mlp_res_proj(outputs_time_res)
            outputs_text = outputs_text + self.mlp_res_w * self.mlp_res_proj(outputs_text_res)
        
        return {
            'outputs_text': outputs_text,
            'outputs_time':outputs_time,
            'intermidiate_time':intermidiate_feat_time,
            'intermidiate_text':intermidiate_feat_text,
        }
    

    def imputation(self, x, mask, cycle_index=None):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        x = x.masked_fill(mask == 0, 0)

        stdev = torch.sqrt(torch.sum(x**2, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5).unsqueeze(1).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x, cycle_index=cycle_index)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
        
        # residue connection
        outputs_time += outputs_time1
        outputs_text += outputs_text1
        
        intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        outputs_time_feat = outputs_time
        outputs_text_feat = outputs_text

        outputs_time = self.out_layer(outputs_time_feat, res_w=self.mlp_res_w)
        outputs_text = self.out_layer(outputs_text_feat, res_w=self.mlp_res_w)


        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time':outputs_time,
            'intermidiate_time':intermidiate_feat_time,
            'intermidiate_text':intermidiate_feat_text,
        }

    def anomaly_detection(self, x, cycle_index=None):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x, cycle_index=cycle_index)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
        # residue connection
        outputs_time += outputs_time1
        outputs_text += outputs_text1
        
        intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        outputs_time_feat = outputs_time
        outputs_text_feat = outputs_text

        outputs_time = self.out_layer(outputs_time_feat, res_w=self.mlp_res_w)
        outputs_text = self.out_layer(outputs_text_feat, res_w=self.mlp_res_w)


        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time':outputs_time,
            'intermidiate_time':intermidiate_feat_time,
            'intermidiate_text':intermidiate_feat_text,
        }


    def forward(self, x, mask=None, cycle_index=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            output = self.forecast(x, cycle_index=cycle_index)
        if self.task_name == 'classification':
            output = self.classification(x, cycle_index=cycle_index)
        if self.task_name == "imputation":
            output = self.imputation(x, mask, cycle_index=cycle_index)
        if self.task_name == "anomaly_detection":
            output = self.anomaly_detection(x, cycle_index=cycle_index)
        return output
