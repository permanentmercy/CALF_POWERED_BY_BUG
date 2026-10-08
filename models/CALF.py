import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import os
import numpy as np
import pandas as pd
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

class MultiScaleTemporalConv(nn.Module):
    """
    轻量多尺度时序卷积 (MTC) 模块：
    沿时间序列维度提取局部趋势与多频段波动特征（k=3, 5, 7 分别对应高频、中频与宏观形态），
    并融合投影至 hidden_dim，作为输入端全局线性投影的有力补充。
    """
    def __init__(self, seq_len, hidden_dim, dropout=0.1):
        super(MultiScaleTemporalConv, self).__init__()
        self.conv3 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.fuse = nn.Conv1d(48, 1, kernel_size=1)
        self.act = nn.GELU()
        self.linear = nn.Linear(seq_len, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, M, seq_len)
        B, M, L = x.shape
        x_in = x.reshape(B * M, 1, L)
        c3 = self.act(self.conv3(x_in))
        c5 = self.act(self.conv5(x_in))
        c7 = self.act(self.conv7(x_in))
        c = self.fuse(torch.cat([c3, c5, c7], dim=1))
        c = c.reshape(B, M, L)
        out = self.dropout(self.linear(c))
        return self.norm(out)

class ResMLPBlock(nn.Module):
    """
    带有 Pre-LayerNorm 和残差连接的 MLP 模块，
    防止多层感知机在深层投影时的梯度退化并加速收敛。
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        return x + self.drop(self.fc2(self.act(self.fc1(self.norm(x)))))

class Encoder_PCA(nn.Module):
    """
    结合主成分分析 (PCA) 词嵌入的编码器模块，
    并包含可选的时间查询 (Temporal Query, TQ) 机制以提取时间序列特征。
    
    该编码器在输入特征与预训练词嵌入之间使用交叉注意力，
    并通过时间查询向量进行增强，以捕获季节性/周期性模式。
    同时支持多尺度时序卷积 (MTC) 增强与时间分支 TQ 相位注入。
    """
    def __init__(self, input_dim, word_embedding, hidden_dim=768, num_heads=12, num_encoder_layers=1, dim_feedforward=2048, cycle_len=24, use_tq_gate=False, tq_dropout=0.0, tq_mode='mul', use_tq=True, tq_inject_target='pca', use_conv_emb=True, use_tq_time=True):
        super(Encoder_PCA, self).__init__()
        self.use_tq_gate = use_tq_gate
        self.tq_mode = tq_mode
        self.use_tq = use_tq
        self.use_conv_emb = use_conv_emb
        self.use_tq_time = use_tq_time
        # 'pca': 周期向量注入到 PCA 的 K/V (w_embed) 侧（原始行为）
        # 'q'  : 周期向量注入到时序特征的 Q 侧
        self.tq_inject_target = tq_inject_target
        self.linear = nn.Linear(input_dim, hidden_dim)

        if self.use_conv_emb:
            self.conv_emb = MultiScaleTemporalConv(input_dim, hidden_dim, dropout=tq_dropout)
            self.conv_gate = nn.Parameter(torch.zeros(1))
        if self.use_tq_time:
            self.tq_time_gate = nn.Parameter(torch.zeros(1))

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
        tq_vec = None
        if self.use_tq and cycle_index is not None:
            tq = self.temporal_query[cycle_index] # (B, 768)
            if self.tq_inject_target == 'pca':
                # 原始行为：调制 PCA 词嵌入 (K/V 侧)
                if self.tq_mode == 'mul':
                    w_embed = w_embed * (torch.sigmoid(self.tq_dropout(tq.unsqueeze(1))) * 2)
                else:
                    w_embed = w_embed + self.tq_dropout(tq.unsqueeze(1))
            else:
                # 'q' 模式：将 TQ 向量缓存，稍后注入时序侧
                tq_vec = self.tq_dropout(tq.unsqueeze(1))  # (B, 1, hidden_dim)

        x_raw = x
        x_lin = self.linear(x_raw)
        if self.use_conv_emb:
            x_time = x_lin + self.conv_gate * self.conv_emb(x_raw)
        else:
            x_time = x_lin

        # 核心架构增强：时间分支感知日周期 TQ 相位（零初始化平滑演进）
        if self.use_tq_time and tq_vec is not None:
            if self.tq_mode == 'mul':
                x_time = x_time * (1.0 + self.tq_time_gate * torch.tanh(tq_vec))
            else:
                x_time = x_time + self.tq_time_gate * tq_vec

        x = x_time

        # 多层特征提取
        for layer in self.layers:
            # Cross-Attention: Query from x, Key/Value from w_embed
            q = x.transpose(0, 1)
            # 'q' 注入模式：在 Q 上叠加周期向量（broadcast 到所有 token）
            if tq_vec is not None:
                if self.tq_mode == 'mul':
                    q = q * (torch.sigmoid(tq_vec.transpose(0, 1)) * 2)
                else:
                    q = q + tq_vec.transpose(0, 1)
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
            # 直接使用 TQ 增强后的特征
            return x_time, x

class TQ_OutputHead(nn.Module):
    """
    专为时间查询 (TQ) 增强架构设计的输出头。
    包含多层感知机 (MLP) 和线性投影层，支持零初始化残差跳跃连接。
    """
    def __init__(self, d_model, output_dim, dropout, out_mlp_layers=2):
        super().__init__()
        layers = []
        for _ in range(out_mlp_layers):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
        self.res_alpha = nn.Parameter(torch.zeros(1))
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, x, res_w=0.0):
        # x: (batch, seq, d_model)
        h = self.mlp(x) + self.res_alpha * x
        if res_w != 0:
            h = h + res_w * x
        return self.output_proj(h)

class Model(nn.Module):
    """
    CALF (用于长期预测的跨模态对齐) 核心架构。
    
    该模型采用双分支结构（时间分支和文本分支），利用预训练的语言模型 (GPT-2)，
    并通过 LoRA 进行微调，以处理时间序列分析任务，包括：
    预测 (forecasting)、分类 (classification)、插补 (imputation) 和异常检测 (anomaly detection)。
    """
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.enc_in = getattr(configs, 'enc_in', None)
        self.c_out = getattr(configs, 'c_out', self.enc_in)
        
        target_modules = ["c_attn"]
        if getattr(configs, 'lora_target_proj', 0):
            target_modules = ["c_attn", "c_proj"]
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=target_modules
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
            use_tq=getattr(configs, 'use_tq', 1),            # 是否开启 TQ 注入
            tq_inject_target=getattr(configs, 'tq_inject_target', 'pca'),  # 注入目标: 'pca'(K/V侧) 或 'q'(Q侧)
            use_conv_emb=bool(getattr(configs, 'use_conv_emb', 1)),
            use_tq_time=bool(getattr(configs, 'use_tq_time', 1))
        )
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.out_layer = TQ_OutputHead(configs.d_model, configs.pred_len, configs.dropout, getattr(configs, 'out_mlp_layers', 2))
        elif self.task_name == 'classification':
            self.out_layer = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
            self.mlp_res_proj = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
        elif self.task_name == 'imputation':
            self.out_layer = TQ_OutputHead(configs.d_model, configs.seq_len, configs.dropout, getattr(configs, 'out_mlp_layers', 2))
        elif self.task_name == 'anomaly_detection':
            self.out_layer = TQ_OutputHead(configs.d_model, configs.seq_len, configs.dropout, getattr(configs, 'out_mlp_layers', 2))
        
        self.mlp_res_w = getattr(configs, 'mlp_res_w', 0.0)

        # Define physical cycle queue parameter if task is forecasting
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.cycle_queue = nn.Parameter(torch.zeros(configs.cycle, configs.enc_in))
            self.cycle_scale = nn.Parameter(torch.ones(1) * 0.1)
            self.cycle_prior_before_denorm = getattr(configs, 'cycle_prior_before_denorm', 1)
            self.cycle_len = getattr(configs, 'cycle', 144)

            # Direct autoregressive linear transition skip connection (ReZero)
            self.direct_trend = nn.Linear(configs.seq_len, configs.pred_len, bias=False)
            nn.init.orthogonal_(self.direct_trend.weight)
            self.trend_gate = nn.Parameter(torch.zeros(1))

            # Diurnal Harmonic Phase Projection (24h, 12h, 8h, 6h harmonics)
            self.harmonic_linear = nn.Linear(8, configs.enc_in, bias=False)
            nn.init.zeros_(self.harmonic_linear.weight)
            self.harmonic_gate = nn.Parameter(torch.zeros(1))

            # Diurnal Phase Gate (DPG) initialized with smooth raised-cosine solar daylight curve
            init_logit = torch.ones(self.cycle_len, 1) * 8.0
            if self.cycle_len == 144:
                smooth_gate = torch.zeros(self.cycle_len, 1)
                smooth_gate[44:101] = 1.0
                t_dawn = torch.arange(28, 44).float()
                smooth_gate[28:44, 0] = 0.5 * (1.0 - torch.cos(np.pi * (t_dawn - 28.0) / 16.0))
                t_dusk = torch.arange(101, 117).float()
                smooth_gate[101:117, 0] = 0.5 * (1.0 + torch.cos(np.pi * (t_dusk - 101.0) / 16.0))
                eps = 1e-4
                clamped = torch.clamp(smooth_gate, eps, 1.0 - eps)
                init_logit = torch.clamp(torch.log(clamped / (1.0 - clamped)), -8.0, 8.0)
            self.diurnal_gate_logit = nn.Parameter(init_logit)

            # Physical Non-negativity Night Floor (raw 0.0 kW in standardized space)
            night_init = torch.full((1, 1, configs.enc_in), -0.713238)
            data_file = os.path.join(getattr(configs, 'root_path', './datasets/Solar/'), getattr(configs, 'data_path', 'solar_AL.txt'))
            if os.path.exists(data_file):
                try:
                    df_raw = pd.read_csv(data_file, header=None)
                    if not np.issubdtype(df_raw.dtypes.iloc[0], np.number):
                        df_raw = df_raw.iloc[:, 1:]
                    train_len = int(len(df_raw) * 0.7)
                    train_vals = df_raw.values[:train_len]
                    means = train_vals.mean(axis=0)
                    stds = np.sqrt(np.var(train_vals, axis=0)) + 1e-5
                    emp_zero = -means / stds
                    if len(emp_zero) >= configs.enc_in:
                        night_init = torch.tensor(emp_zero[:configs.enc_in], dtype=torch.float32).view(1, 1, -1)
                        print(f">>> [CALF Diurnal Prior] Initialized station night_floor for {configs.enc_in} channels.")
                except Exception as e:
                    print(f">>> [CALF Diurnal Prior] Fallback to default night floor: {e}")
            self.night_floor = nn.Parameter(night_init)
        self.use_norm = getattr(configs, 'use_norm', 1)

        for layer in (self.gpt2_text, self.gpt2, self.in_layer, self.out_layer, self.time_proj, self.text_proj):
            layer.to(device=device)
            layer.train()
        if hasattr(self, 'direct_trend'):
            self.direct_trend.to(device=device)
            self.direct_trend.train()
        if hasattr(self, 'harmonic_linear'):
            self.harmonic_linear.to(device=device)
            self.harmonic_linear.train()
        self.to(device=device)
        
        self.cnt = 0
        

    def forecast(self, x, cycle_index=None, future_cycle_index=None):
        """
        执行长期或短期时间序列预测任务。
        
        参数:
            x (Tensor): 输入的时间序列数据，形状为 (Batch, Seq_len, Features)。
            cycle_index (Tensor, 可选): 当前时间周期的索引张量。
            future_cycle_index (Tensor, 可选): 目标未来周期的索引张量。
            
        返回:
            dict: 包含时间分支和文本分支的预测结果，以及它们的中间特征。
        """
        # Auto-slice input channels if data has more channels than target enc_in / cycle_queue
        if hasattr(self, 'cycle_queue') and x.shape[-1] > self.cycle_queue.shape[1]:
            x = x[:, :, :self.cycle_queue.shape[1]]
        elif hasattr(self, 'enc_in') and self.enc_in is not None and x.shape[-1] > self.enc_in:
            x = x[:, :, :self.enc_in]

        B, L, M = x.shape

        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() 
            x /= stdev
        x_norm = x

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

        # Direct autoregressive linear trend skip in normalized space (ReZero)
        if hasattr(self, 'direct_trend') and hasattr(self, 'trend_gate'):
            trend = self.direct_trend(x_norm.transpose(1, 2)).transpose(1, 2)
            outputs_time = outputs_time + self.trend_gate * trend
            outputs_text = outputs_text + self.trend_gate * trend

        # Add physical cycle residual prior if available (before denormalization option)
        if future_cycle_index is not None and hasattr(self, 'cycle_queue') and getattr(self, 'cycle_prior_before_denorm', 1) == 1:
            future_cycle_val = self.cycle_queue[future_cycle_index.to(self.cycle_queue.device)] # (B, pred_len, M)
            outputs_time = outputs_time + self.cycle_scale * future_cycle_val
            outputs_text = outputs_text + self.cycle_scale * future_cycle_val

        if self.use_norm:
            outputs_text = outputs_text * stdev + means
            outputs_time = outputs_time * stdev + means

        # Add physical cycle residual prior if available (after denormalization option)
        if future_cycle_index is not None and hasattr(self, 'cycle_queue') and getattr(self, 'cycle_prior_before_denorm', 1) == 0:
            future_cycle_val = self.cycle_queue[future_cycle_index.to(self.cycle_queue.device)] # (B, pred_len, M)
            outputs_time = outputs_time + self.cycle_scale * future_cycle_val
            outputs_text = outputs_text + self.cycle_scale * future_cycle_val

        # Diurnal Harmonic Phase Projection and Diurnal Phase Gate (DPG) with Physical Bound
        if future_cycle_index is not None and hasattr(self, 'diurnal_gate_logit') and hasattr(self, 'night_floor'):
            fc_idx = future_cycle_index.to(self.diurnal_gate_logit.device) # (B, pred_len)
            
            # Harmonic Phase Projection
            if hasattr(self, 'harmonic_linear') and hasattr(self, 'harmonic_gate'):
                cycle_len = getattr(self, 'cycle_len', 144)
                phi = 2.0 * np.pi * fc_idx.float() / cycle_len
                harmonics = torch.stack([
                    torch.sin(phi), torch.cos(phi),
                    torch.sin(2.0 * phi), torch.cos(2.0 * phi),
                    torch.sin(3.0 * phi), torch.cos(3.0 * phi),
                    torch.sin(4.0 * phi), torch.cos(4.0 * phi)
                ], dim=-1)
                h_out = self.harmonic_linear(harmonics) # (B, pred_len, M)
                outputs_time = outputs_time + self.harmonic_gate * h_out
                outputs_text = outputs_text + self.harmonic_gate * h_out

            # Diurnal Phase Gate (gate -> 0 at night, gate -> 1 during day)
            gate = torch.sigmoid(self.diurnal_gate_logit[fc_idx]) # (B, pred_len, 1)
            
            # Soft Non-negativity Bound (smooth softplus relaxation to eliminate hard step clamping)
            time_diff = outputs_time - self.night_floor
            text_diff = outputs_text - self.night_floor
            outputs_time_phys = self.night_floor + F.softplus(time_diff, beta=10.0)
            outputs_text_phys = self.night_floor + F.softplus(text_diff, beta=10.0)
            
            # Smooth diurnal gating: nighttime smoothly clamps to night_floor, daytime retains neural forecast
            outputs_time = gate * outputs_time_phys + (1.0 - gate) * self.night_floor
            outputs_text = gate * outputs_text_phys + (1.0 - gate) * self.night_floor

        return {
            'outputs_text': outputs_text,
            'outputs_time':outputs_time,
            'intermidiate_time':intermidiate_feat_time,
            'intermidiate_text':intermidiate_feat_text,
        }


    def classification(self, x, cycle_index=None):
        """
        执行时间序列分类任务。
        
        参数:
            x (Tensor): 输入的时间序列数据。
            cycle_index (Tensor, 可选): 用于时间查询的索引张量。
            
        返回:
            dict: 各个分支的分类预测结果以及中间特征。
        """
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
        """
        执行时间序列缺失值插补任务。
        
        参数:
            x (Tensor): 包含缺失值的输入时间序列数据。
            mask (Tensor): 布尔掩码，指示观测值 (1) 和缺失值 (0)。
            cycle_index (Tensor, 可选): 用于时间查询的索引张量。
            
        返回:
            dict: 重构后的时间序列数据以及中间特征。
        """
        B, L, M = x.shape

        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            x = x.masked_fill(mask == 0, 0)
            stdev = torch.sqrt(torch.sum(x**2, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5).unsqueeze(1).detach()
            x /= stdev
        else:
            x = x.masked_fill(mask == 0, 0)

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

        if self.use_norm:
            outputs_text = outputs_text * stdev + means
            outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time':outputs_time,
            'intermidiate_time':intermidiate_feat_time,
            'intermidiate_text':intermidiate_feat_text,
        }

    def anomaly_detection(self, x, cycle_index=None):
        """
        执行时间序列异常检测任务（通过尝试重构输入数据来实现）。
        
        参数:
            x (Tensor): 输入的时间序列数据。
            cycle_index (Tensor, 可选): 用于时间查询的索引张量。
            
        返回:
            dict: 重构后的时间序列（用于与原始数据比较计算异常分数）以及中间特征。
        """
        B, L, M = x.shape

        if self.use_norm:
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

        if self.use_norm:
            outputs_text = outputs_text * stdev + means
            outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time':outputs_time,
            'intermidiate_time':intermidiate_feat_time,
            'intermidiate_text':intermidiate_feat_text,
        }


    def forward(self, x, mask=None, cycle_index=None, future_cycle_index=None):
        """
        前向传播函数，根据配置中的任务名称路由到对应的特定任务方法。
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            output = self.forecast(x, cycle_index=cycle_index, future_cycle_index=future_cycle_index)
        if self.task_name == 'classification':
            output = self.classification(x, cycle_index=cycle_index)
        if self.task_name == "imputation":
            output = self.imputation(x, mask, cycle_index=cycle_index)
        if self.task_name == "anomaly_detection":
            output = self.anomaly_detection(x, cycle_index=cycle_index)
        return output

    def load_state_dict(self, state_dict, strict=True):
        """
        自适应权重加载方法：
        当模型与预训练权重中的 cycle_queue 通道数 (enc_in) 不一致时：
        1. 若目标通道数 < 预训练通道数 (如 27 < 127 或 29 < 137)：自动切片截取前 target_enc_in 列，并提示；
        2. 若目标通道数 > 预训练通道数 (如 137 > 29)：主动抛出 ValueError 错误提示；
        3. 若通道数一致：正常直接加载。
        """
        if 'cycle_queue' in state_dict and hasattr(self, 'cycle_queue'):
            ckpt_enc_in = state_dict['cycle_queue'].shape[1]
            curr_enc_in = self.cycle_queue.shape[1]
            if ckpt_enc_in != curr_enc_in:
                if curr_enc_in < ckpt_enc_in:
                    print(f">>> [Adaptive Load] Slicing cycle_queue from checkpoint channels ({ckpt_enc_in}) to target model channels ({curr_enc_in}).")
                    state_dict = state_dict.copy()
                    state_dict['cycle_queue'] = state_dict['cycle_queue'][:, :curr_enc_in]
                else:
                    raise ValueError(
                        f"\n[Adaptive Load Error] Insufficient checkpoint channels!\n"
                        f"Checkpoint cycle_queue has {ckpt_enc_in} channels, but target model requires {curr_enc_in} channels ({curr_enc_in} > {ckpt_enc_in}).\n"
                        f"Cannot expand fewer channels to larger channels without training."
                    )
        if 'cycle_scale' in state_dict and hasattr(self, 'cycle_scale'):
            ckpt_scale = state_dict['cycle_scale']
            if ckpt_scale.shape != self.cycle_scale.shape:
                state_dict = state_dict.copy()
                if ckpt_scale.numel() == 1:
                    state_dict['cycle_scale'] = self.cycle_scale.clone().fill_(ckpt_scale.item())
                elif ckpt_scale.shape[-1] != self.cycle_scale.shape[-1]:
                    min_c = min(ckpt_scale.shape[-1], self.cycle_scale.shape[-1])
                    new_scale = self.cycle_scale.clone()
                    new_scale[:, :, :min_c] = ckpt_scale[:, :, :min_c]
                    state_dict['cycle_scale'] = new_scale
        return super().load_state_dict(state_dict, strict=strict)

