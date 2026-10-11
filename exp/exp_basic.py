import os
import torch
from models import CALF


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "CALF": CALF,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        # ── 训练集百分比缩减与权重加载控制 ──────────────────────────
        # 场景 1: 少样本微调 (percent < 100 且指定了 checkpoint_path)
        # 场景 2: 少样本从头训练 (percent < 100 且未指定 checkpoint_path)
        is_training = (getattr(args, 'is_training', 0) == 1)
        percent = getattr(args, 'percent', 100)
        ckpt = getattr(args, 'checkpoint_path', '0')
        has_ckpt = ckpt and str(ckpt).strip() not in ['0', 'None', '']

        if is_training and percent < 100:
            if has_ckpt:
                # 显式指定了 checkpoint，执行预训练权重加载与少样本微调
                if os.path.isdir(ckpt):
                    ckpt = os.path.join(ckpt, 'checkpoint.pth')
                if not os.path.isfile(ckpt):
                    raise FileNotFoundError(
                        f"[Few-Shot] 指定的预训练权重文件不存在: {ckpt}"
                    )
                print(f"[Few-Shot Fine-Tuning] 正在加载预训练权重: {ckpt}")
                state = torch.load(ckpt, map_location=self.device)
                missing, unexpected = self.model.load_state_dict(state, strict=False)
                if missing:
                    print(f"[Few-Shot Fine-Tuning] 以下参数在 checkpoint 中缺失（将使用随机初始化）: {missing}")
                if unexpected:
                    print(f"[Few-Shot Fine-Tuning] checkpoint 中存在当前模型不需要的参数（已忽略）: {unexpected}")
                print(f"[Few-Shot Fine-Tuning] 权重加载完成，将使用 {percent:g}% 数据进行微调。")
            else:
                # 未指定 checkpoint，使用 percent% 数据从头训练全新的模型
                print(f"[Few-Shot Scratch] 从头训练全新模型：使用前 {percent:g}% 的训练数据进行端到端优化（无预训练权重）。")

            # ── 可选：冻结 GPT2 主干，只训练投影层和输出头 ──────────────
            if getattr(args, 'freeze_backbone', False):
                frozen_count = 0
                for name, param in self.model.named_parameters():
                    if 'gpt2' in name:
                        param.requires_grad = False
                        frozen_count += param.numel()
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print(f"[Few-Shot] GPT2 主干已冻结（冻结参数量: {frozen_count:,}），"
                      f"可训练参数: {trainable:,}")
        # ────────────────────────────────────────────────────────────────

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
