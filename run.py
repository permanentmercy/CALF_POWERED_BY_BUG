import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    # fix_seed = 2021
    # random.seed(fix_seed)
    # torch.manual_seed(fix_seed)
    # np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')
    
    parser.add_argument('--random_seed', type=int, default=2023, help='random seed')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    
    parser.add_argument('--task_loss', type=str, default='l1', help='task loss function')
    parser.add_argument('--feature_loss', type=str, default='l1', help='distillation loss function')
    parser.add_argument('--output_loss', type=str, default='l1', help='logits loss function')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # self
    parser.add_argument('--tmax', type=int, default=20) # rl 余弦调整
    parser.add_argument('--cos', type=int, default=1)

    # lora
    parser.add_argument('--r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)

    # align
    parser.add_argument('--word_embedding_path', type=str, default="wte_pca_500.pt")

    # loss weight
    parser.add_argument('--task_w', type=float, default=1.0) 
    parser.add_argument('--feature_w', type=float, default=0.01) # 
    parser.add_argument('--output_w', type=float, default=1.0)
    # save best model in memory instead of (or as well as) disk
    parser.add_argument('--bestmodel', action='store_true', help='save best model in memory and load from it during test', default=False)
    
    # gpt
    parser.add_argument('--gpt_layers', type=int, default=32, help='number of hidden layers in gpt')
    parser.add_argument('--gpt2_path', type=str, default=None, help='path or huggingface id for GPT-2 model (optional)')
    
    # training data percentage
    parser.add_argument('--percent', type=int, default=100,
                        help='使用的训练数据百分比（1-100）。默认100（全量从头训练）。'
                             '仅在 is_training=1 时控制训练集大小。'
                             '当 percent<100 时截取训练集前 percent% 的数据从头训练全新模型；'
                             '若同时通过 --checkpoint_path 指定预训练权重，则执行少样本微调。')
    
    # zero shot
    parser.add_argument('--zero_shot', type=int, default=0, help='1 for training zero shot')
    parser.add_argument('--target_data', type=str, default="ETTm1", help='target data for zero shot')

    # few-shot fine-tuning options
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                        help='少样本微调时冻结 GPT2 主干（gpt2/gpt2_text），只训练投影层和输出头，'
                             '防止在极少数据下过拟合。仅在 is_training=1, percent<100 时生效。')

    # evaluation
    parser.add_argument('--eval_test_every_epoch', action='store_true', help='evaluate test set every epoch during training', default=False)

    # TQNet cycle
    parser.add_argument('--cycle', type=int, default=24, help='cycle length for TQ mechanism')
    parser.add_argument('--use_tq_gate', type=int, default=0, help='whether to use learnable gate for TQ injection')
    parser.add_argument('--use_swa', type=int, default=0, help='whether to use SWA')
    parser.add_argument('--tq_lr_factor', type=float, default=10.0, help='learning rate factor for TQ related parameters')
    parser.add_argument('--tq_dropout', type=float, default=0.0, help='dropout probability for TQ injection')
    parser.add_argument('--tq_mode', type=str, default='mul', choices=['add', 'mul'], help='TQ injection mode: add or mul')
    parser.add_argument('--tq_inject_target', type=str, default='pca', choices=['pca', 'q'],
                        help='where to inject the cycle feature vector: "pca" injects into PCA K/V (original), "q" injects into temporal Q')
    parser.add_argument('--layer_offset', type=int, default=0, help='offset for time branch layer indexing in feature alignment')
    parser.add_argument('--use_tq', type=int, default=1, help='enable or disable TQ injection completely')
    parser.add_argument('--test_branch', type=str, default='time', choices=['time', 'text'], help='which branch to use for testing/validation')
    parser.add_argument('--mlp_res_w', type=float, default=0.0, help='weight of the residual connection after MLP')
    parser.add_argument('--out_mlp_layers', type=int, default=2, help='number of MLP layers in the output head')
    parser.add_argument('--use_ef', action='store_true', default=False,
                        help='Enable Evolutionary Forecasting (EF) inference: autoregressive block extrapolation at test time for horizons 96/192/336/720')
    parser.add_argument('--cycle_prior_before_denorm', type=int, default=1,
                        help='whether to add physical cycle prior before denormalization (1: before, 0: after)')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use internal normalization (RevIN)')
    parser.add_argument('--use_conv_emb', type=int, default=0, help='enable Multi-Scale Temporal Conv in Encoder_PCA')
    parser.add_argument('--use_tq_time', type=int, default=0, help='inject TQ vector into time branch x_time')
    parser.add_argument('--task_time_ratio', type=float, default=0.5, help='weight ratio for time branch in supervised task loss')
    parser.add_argument('--lora_target_proj', type=int, default=0, help='also adapt c_proj in GPT2 with LoRA')
    parser.add_argument('--checkpoint_path', type=str, default="0",
                        help='权重文件路径（.pth 文件或含 checkpoint.pth 的目录）。'
                             '生效场景：'
                             '(1) 少样本微调（is_training=1, percent<100）：作为微调起点预训练权重；'
                             '(2) 零样本评估（is_training=0, zero_shot=1）：直接用此权重推理；'
                             '(3) 纯测试（is_training=0）：加载此权重做测试。'
                             '普通全量训练（is_training=1, percent=100）时此参数被忽略。'
                             '若为 "0" 则使用默认 setting 路径。')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    # ── 实验模式检测 ─────────────────────────────────────────────────
    _ckpt = str(getattr(args, 'checkpoint_path', '0')).strip()
    _ckpt_valid = _ckpt not in ['0', 'None', '']

    _is_training = (args.is_training == 1)
    _is_zero_shot = (args.is_training == 0 and args.zero_shot == 1)
    _is_test_only = (args.is_training == 0 and args.zero_shot == 0)

    if _is_training:
        if _ckpt_valid:
            print(f"[Mode] Fine-Tuning | percent={args.percent}% | ckpt={args.checkpoint_path}")
        else:
            if args.percent < 100:
                print(f"[Mode] Training from Scratch | percent={args.percent}% (using {args.percent}% of training data to train a new model)")
            else:
                print("[Mode] Full Training from Scratch | percent=100%")
    elif _is_zero_shot:
        print(f"[Mode] Zero-Shot Evaluation | target_data={args.target_data} | ckpt={args.checkpoint_path}")
    elif _is_test_only:
        print(f"[Mode] Test-Only | ckpt={args.checkpoint_path if _ckpt_valid else '(default setting path)'}")
    # ─────────────────────────────────────────────────────────────────

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_gpt{}_rl{}_{}_{}_feature{}_output{}_train_epochs{}_bs{}_dr{}_ld{}_r{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, 
                args.gpt_layers,
                args.learning_rate,
                args.random_seed,
                ii,
                args.feature_w,
                args.output_w,
                args.train_epochs,
                args.batch_size,
                args.dropout,
                args.lora_dropout,
                args.r)

            # 部分数据训练/少样本模式：在 setting 末尾追加 _p{percent} 标识，
            # 确保 checkpoint 和输出目录独立，不会覆盖全量训练的权重。
            if args.percent < 100:
                setting += f'_p{args.percent}'

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_gpt{}_rl{}_{}_{}_feature{}_output{}_train_epochs{}_bs{}_dr{}_ld{}_r{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, 
            args.gpt_layers,
            args.learning_rate,
            args.random_seed,
            ii,
            args.feature_w,
            args.output_w,
            args.train_epochs,
            args.batch_size,
            args.dropout,
            args.lora_dropout,
            args.r)

        if args.percent < 100:
            setting += f'_p{args.percent}'

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()
