import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, input_dim, word_embedding, hidden_dim=768, num_heads=12, num_encoder_layers=1, dim_feedforward=2048):
        super(Encoder_PCA, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

        # 让脚本里的 d_ff（dim_feedforward）真正影响 Transformer 前馈网络维度
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,                 # 注意力层的输入维度
            nhead=num_heads,                   # 多头注意力头数
            dim_feedforward=dim_feedforward,  # Transformer FFN 中间维度（由 d_ff 控制）
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        
        self.word_embedding = word_embedding.T

    def forward(self, x):
        B = x.shape[0]
        if self.word_embedding.ndim == 2:
            self.word_embedding = self.word_embedding.repeat(B, 1, 1)
        elif self.word_embedding.shape[0] != B:
            self.word_embedding = self.word_embedding[0].repeat(B, 1, 1)

        x = self.linear(x)

        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)

        x_time = x

        q = x.transpose(0, 1)
        k = v = self.word_embedding.transpose(0, 1)
        x, _ = self.cross_attention(q, k, v)

        x = x.transpose(0, 1)

        return x_time, x

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

        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]#通过gpt_layers裁剪层数
        self.gpt2_text.h = self.gpt2_text.h[:configs.gpt_layers]
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

        # time→self-attn→cross-attn→pos_enc→残差到文本层最后自注意力
        self.time_to_text_self_attn = nn.MultiheadAttention(
            embed_dim=configs.d_model, num_heads=configs.n_heads, batch_first=True, dropout=configs.dropout
        )
        self.time_to_text_cross_attn = nn.MultiheadAttention(
            embed_dim=configs.d_model, num_heads=configs.n_heads, batch_first=True, dropout=configs.dropout
        )
        self.time_to_text_ln = nn.LayerNorm(configs.d_model)
        self.time_to_text_dropout = nn.Dropout(configs.dropout)
        # 可学习的残差缩放系数，初始化为较小值防止过拟合
        self.time_to_text_scale = nn.Parameter(torch.tensor(0.1))

        # 新增：从文本层输入到时间层输出的变换链路
        self.text_to_time_link = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.LayerNorm(configs.d_model)
        )
        self.text_to_time_dropout = nn.Dropout(configs.gate_dropout)
        
        # 门控标量初始化为 -3.0，使初始 sigmoid(gate) ≈ 0.047
        self.text_to_time_gate = nn.Parameter(torch.tensor(-3.0))
        
        # 恒等映射初始化：将线性层权重和偏置置零，确保初期注入几乎为 0
        nn.init.zeros_(self.text_to_time_link[0].weight)
        nn.init.zeros_(self.text_to_time_link[0].bias)

        # 把 configs.d_ff 传进 Encoder_PCA，让 --d_ff 生效
        self.in_layer = Encoder_PCA(
            configs.seq_len,                  # 输入维度（线性层输入）
            word_embedding,                  # 用于交叉注意力的词嵌入
            hidden_dim=configs.d_model,    # Transformer 的 d_model
            dim_feedforward=configs.d_ff,  # Transformer FFN 中间维度（来自脚本）
        )
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.out_layer = nn.Linear(configs.d_model, configs.pred_len)
        elif self.task_name == 'classification':
            self.out_layer = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
        elif self.task_name == 'imputation':
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)
        elif self.task_name == 'anomaly_detection':
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)

        for layer in (self.gpt2_text, self.gpt2, self.in_layer, self.out_layer, self.time_proj, self.text_proj,
                       self.time_to_text_self_attn, self.time_to_text_cross_attn, self.time_to_text_ln,
                       self.time_to_text_dropout, self.text_to_time_dropout):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0
        

    def forecast(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x)

        # ---- time→self-attn→cross-attn(与text)→pos_enc→残差到文本层最后自注意力 ----
        time_self, _ = self.time_to_text_self_attn(outputs_time1, outputs_time1, outputs_time1)
        time_cross, _ = self.time_to_text_cross_attn(time_self, outputs_text1, outputs_text1)
        seq_len_t = time_cross.shape[1]
        position_ids = torch.arange(seq_len_t, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_embeds = self.gpt2_text.wpe(position_ids)
        time_residual = time_cross + pos_embeds
        time_residual = self.time_to_text_ln(time_residual)
        time_residual = self.time_to_text_dropout(time_residual)
        time_residual = time_residual * self.time_to_text_scale
        # ----

        # 获取文本层经过位置编码后的初始信息
        pos_ids = torch.arange(M, dtype=torch.long, device=x.device).unsqueeze(0)
        text_with_pos = outputs_text1 + self.gpt2_text.wpe(pos_ids)
        
        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        
        # 1. 计算门控值
        gate = torch.sigmoid(self.text_to_time_gate)
        text_bias = self.text_to_time_link(text_with_pos)
        text_bias = self.text_to_time_dropout(text_bias)

        # 2. 诊断信号：计算相加前后的余弦相似度（监控融合剧烈程度）
        with torch.no_grad():
            cos_sim = F.cosine_similarity(outputs_time.reshape(B, -1), 
                                          (outputs_time + gate * text_bias).reshape(B, -1)).mean()
            bias_norm = torch.norm(gate * text_bias)

        # 3. 带门控的残差注入
        outputs_time = outputs_time + gate * text_bias

        outputs_text, intermidiate_feat_text = self.gpt2_text(
            inputs_embeds=outputs_text1, external_residual=time_residual
        )
        # residue connection
        outputs_time += outputs_time1
        outputs_text += outputs_text1
        
        intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        outputs_time = self.out_layer(outputs_time[:, -M:, :])
        outputs_text = self.out_layer(outputs_text[:, -M:, :])

        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time': outputs_time,
            'intermidiate_time': intermidiate_feat_time,
            'intermidiate_text': intermidiate_feat_text,
            'gate_value': gate.item(),
            'cos_sim': cos_sim.item(),
            'bias_norm': bias_norm.item()
        }


    def classification(self, x):
        B, L, M = x.shape

        x = rearrange(x, 'b l m -> b m l')

        # 获取文本层经过位置编码后的初始信息
        pos_ids = torch.arange(M, dtype=torch.long, device=x.device).unsqueeze(0)
        text_with_pos = outputs_text1 + self.gpt2_text.wpe(pos_ids)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        
        # 加入变换后的文本信息
        outputs_time = outputs_time + self.text_to_time_link(text_with_pos)

        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
        
        outputs_time += outputs_time1
        outputs_text += outputs_text1
        
        intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])
        
        outputs_time = outputs_time.reshape(B, -1)
        outputs_text = outputs_text.reshape(B, -1)
        
        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)
        
        return {
            'outputs_text': outputs_text,
            'outputs_time':outputs_time,
            'intermidiate_time':intermidiate_feat_time,
            'intermidiate_text':intermidiate_feat_text,
        }
    

    def imputation(self, x, mask):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        x = x.masked_fill(mask == 0, 0)

        stdev = torch.sqrt(torch.sum(x**2, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5).unsqueeze(1).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x)

        # 获取文本层经过位置编码后的初始信息
        pos_ids = torch.arange(M, dtype=torch.long, device=x.device).unsqueeze(0)
        text_with_pos = outputs_text1 + self.gpt2_text.wpe(pos_ids)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        
        # 加入变换后的文本信息
        outputs_time = outputs_time + self.text_to_time_link(text_with_pos)

        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
        
        # residue connection
        outputs_time += outputs_time1
        outputs_text += outputs_text1
        
        intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)

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

    def anomaly_detection(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x)

        # 获取文本层经过位置编码后的初始信息
        pos_ids = torch.arange(M, dtype=torch.long, device=x.device).unsqueeze(0)
        text_with_pos = outputs_text1 + self.gpt2_text.wpe(pos_ids)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        
        # 加入变换后的文本信息
        outputs_time = outputs_time + self.text_to_time_link(text_with_pos)

        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
        
        # residue connection
        outputs_time += outputs_time1
        outputs_text += outputs_text1
        
        intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)

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


    def forward(self, x, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            output = self.forecast(x)
        if self.task_name == 'classification':
            output = self.classification(x)
        if self.task_name == "imputation":
            output = self.imputation(x, mask)
        if self.task_name == "anomaly_detection":
            output = self.anomaly_detection(x)
        return output
