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

        # ── 少样本模式：加载预训练权重后再微调 ──────────────────────────
        # 触发条件：is_training=1（训练模式）且 percent<100（使用部分数据）
        # 其他情况（普通全量训练）checkpoint_path 完全无效
        is_few_shot = (
            getattr(args, 'is_training', 0) == 1
            and getattr(args, 'percent', 100) < 100
        )
        if is_few_shot:
            ckpt = getattr(args, 'checkpoint_path', '0')
            if not ckpt or str(ckpt).strip() in ['0', 'None', '']:
                raise ValueError(
                    "[Few-Shot] 少样本模式（percent<100）必须通过 --checkpoint_path 指定预训练权重路径，"
                    "当前值为空或 '0'。"
                )
            # 支持传入目录（自动寻找 checkpoint.pth）或直接文件路径
            if os.path.isdir(ckpt):
                ckpt = os.path.join(ckpt, 'checkpoint.pth')
            if not os.path.isfile(ckpt):
                raise FileNotFoundError(
                    f"[Few-Shot] 指定的预训练权重文件不存在: {ckpt}"
                )
            print(f"[Few-Shot] 正在加载预训练权重: {ckpt}")
            state = torch.load(ckpt, map_location=self.device)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing:
                print(f"[Few-Shot] 以下参数在 checkpoint 中缺失（将使用随机初始化）: {missing}")
            if unexpected:
                print(f"[Few-Shot] checkpoint 中存在当前模型不需要的参数（已忽略）: {unexpected}")
            print(f"[Few-Shot] 权重加载完成，将使用 {args.percent}% 数据进行微调。")

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
