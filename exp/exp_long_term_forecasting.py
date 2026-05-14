from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual,test_params_flop
from utils.metrics import metric
from utils.cmLoss import cmLoss
import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import autocast, GradScaler
import os
import time
import warnings
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
import pandas as pd
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.best_model_state = None
        self.best_model_epoch = None
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args, self.device).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        # 打印模型配置诊断信息
        print("\n" + "="*60)
        print("MODEL CONFIGURATION DIAGNOSTIC")
        print("="*60)
        print(f"gpt_layers parameter: {self.args.gpt_layers}")
        
        # 获取实际模型（处理 DataParallel 的情况）
        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        
        # 打印 GPT2 层数信息
        if hasattr(actual_model, 'gpt2'):
            print(f"actual gpt2.h length: {len(actual_model.gpt2.h)}")
            print(f"config.gpt_layers: {actual_model.gpt2.config.n_layer}")
        
        # 打印模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # 打印 time_proj 和 text_proj 的长度
        if hasattr(actual_model, 'time_proj'):
            print(f"time_proj length: {len(actual_model.time_proj)}")
        if hasattr(actual_model, 'text_proj'):
            print(f"text_proj length: {len(actual_model.text_proj)}")
        print("="*60 + "\n")
        
        return model

    def _get_data(self, flag, vali_test=False):
        data_set, data_loader = data_provider(self.args, flag, vali_test)
        return data_set, data_loader

    def _select_optimizer(self):
        # 参数分组：
        # 1. 门控参数 (text_to_time_gate) - 使用缩减后的学习率
        # 2. 投影层参数 (_proj) - 使用标准学习率
        # 3. 其他所有参数 - 使用标准学习率
        gate_params = [p for n, p in self.model.named_parameters() if p.requires_grad and 'text_to_time_gate' in n]
        proj_params = [p for n, p in self.model.named_parameters() if p.requires_grad and '_proj' in n]
        
        # 排除掉上述已经分类的参数，剩下的作为 other_params
        gate_ids = set(map(id, gate_params))
        proj_ids = set(map(id, proj_params))
        other_params = [p for p in self.model.parameters() if p.requires_grad and id(p) not in gate_ids and id(p) not in proj_ids]
        
        param_groups = [
            {"params": other_params, "lr": self.args.learning_rate, "lr_factor": 1.0},
            {"params": proj_params, "lr": self.args.learning_rate * 2.0, "lr_factor": 2.0}, # 恢复双倍更新的强度
            {"params": gate_params, "lr": self.args.learning_rate * self.args.gate_lr_factor, "lr_factor": self.args.gate_lr_factor} # 正确使用 gate_lr_factor
        ]
        # 过滤掉空的参数组，防止优化器报错
        param_groups = [pg for pg in param_groups if len(pg['params']) > 0]
        
        model_optim = optim.Adam(param_groups)
        return model_optim

    def _select_criterion(self):
        criterion = cmLoss(self.args.feature_loss, 
                           self.args.output_loss, 
                           self.args.task_loss, 
                           self.args.task_name, 
                           self.args.feature_w, 
                           self.args.output_w, 
                           self.args.task_w)
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test', vali_test=True)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        # Initialize GradScaler for AMP
        scaler = GradScaler() if self.args.use_amp else None
        if self.args.use_amp:
            print("Mixed Precision Training (AMP) Enabled")
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        
        if self.args.use_swa:
            swa_model = None
            swa_scheduler = None
            swa_active = False
            print(f"SWA enabled in auto-trigger mode (waiting for validation plateau)...")
        
        epoch_times = []
        best_val = np.Inf
        
        accumulation_steps = self.args.accumulation_steps
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            max_memory = 0
            
            # Running totals for logging/averaging
            running_loss = 0.0
            running_task_loss = 0.0
            
            model_optim.zero_grad() # Ensure gradients are zeroed at start of epoch
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # Forward pass with AMP if enabled
                if self.args.use_amp:
                    with autocast():
                        outputs_dict = self.model(batch_x)
                        loss, task_loss, output_loss, feature_loss = criterion(outputs_dict, batch_y)
                else:
                    outputs_dict = self.model(batch_x)
                    loss, task_loss, output_loss, feature_loss = criterion(outputs_dict, batch_y)

                train_loss.append(loss.item())
                running_loss += loss.item()
                running_task_loss += task_loss.item()

                # Backward pass with AMP if enabled
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Optimizer step with gradient accumulation
                if (i + 1) % accumulation_steps == 0 or (i + 1) == train_steps:
                    if self.args.use_amp:
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        model_optim.step()
                    model_optim.zero_grad()

                if (i + 1) % 100 == 0:
                    # 打印门控状态和诊断信号
                    gate_val = outputs_dict.get('gate_value', 0)
                    cos_sim = outputs_dict.get('cos_sim', 0)
                    
                    # Compute average loss over the logging interval (100 steps)
                    avg_loss = running_loss / 100
                    avg_task_loss_log = running_task_loss / 100
                    
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | task: {3:.7f} | Gate: {4:.4f} | CosSim: {5:.4f}".format(
                        i + 1, epoch + 1, avg_loss, avg_task_loss_log, gate_val, cos_sim))
                    
                    running_loss = 0.0
                    running_task_loss = 0.0
                    
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                current_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                max_memory = max(max_memory, current_memory)
            
            t = time.time() - epoch_time
            print("Epoch: {} cost time: {:4f} s".format(epoch + 1, t))
            epoch_times.append(t)
            
            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            
            # Evaluate test set only if eval_test_every_epoch is True
            if getattr(self.args, 'eval_test_every_epoch', False):
                test_loss = self.vali(test_data, test_loader, criterion)
            else:
                test_loss = None

            # 打印各分量损失
            avg_task_loss = task_loss.item() if isinstance(task_loss, torch.Tensor) else 0
            avg_output_loss = output_loss.item() if isinstance(output_loss, torch.Tensor) else 0
            avg_feature_loss = feature_loss.item() if isinstance(feature_loss, torch.Tensor) else 0

            # save best model (to memory or disk)
            if vali_loss < best_val:
                best_val = vali_loss
                self.best_model_epoch = epoch + 1
                if getattr(self.args, 'bestmodel', False):
                    # keep best model in memory
                    self.best_model_state = deepcopy(self.model.state_dict())
                    if self.args.patience and self.args.patience > 0:
                        if early_stopping.verbose:
                            print(f"Best model stored in memory (Vali Loss: {best_val:.6f})")
                else:
                    # save to disk (fallback)
                    try:
                        torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
                        if early_stopping.verbose:
                            print(f"Best model saved to disk: {path}/checkpoint.pth (Vali Loss: {best_val:.6f})")
                    except Exception:
                        pass

            # Print training progress with or without test loss
            if test_loss is not None:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f} | task:{5:.6f} output:{6:.6f} feature:{7:.6f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss,
                    avg_task_loss, avg_output_loss, avg_feature_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} | task:{4:.6f} output:{5:.6f} feature:{6:.6f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss,
                    avg_task_loss, avg_output_loss, avg_feature_loss))

            # 先执行 EarlyStopping 判定，获取最新的 counter
            early_stopping(vali_loss, self.model, path)

            if self.args.lradj == 'TST':
                adjust_learning_rate(model_optim, epoch + 1, self.args)
                scheduler.step()
            else:
                if self.args.use_swa:
                    # 动态 SWA 逻辑：如果找到更好的模型，则重置 SWA；如果开始停滞，则开启/继续 SWA
                    if early_stopping.counter == 0:
                        if swa_active:
                            print(f"\n>>> New best model found at Epoch {epoch + 1}! Resetting SWA average...")
                            # 显存回收，确保在模型较大时不发生 OOM
                            del swa_model
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        swa_active = False
                        swa_model = None
                        swa_scheduler = None
                    
                    if not swa_active and (
                        (self.args.swa_start_epoch != -1 and epoch >= self.args.swa_start_epoch) or \
                        (self.args.swa_start_epoch == -1 and early_stopping.counter >= 1)
                    ):
                        swa_active = True
                        swa_model = AveragedModel(self.model)
                        swa_scheduler = SWALR(model_optim, swa_lr=self.args.learning_rate)
                        print(f"\n>>> SWA (Re)Triggered at Epoch {epoch + 1}! Starting weight averaging using global LR {self.args.learning_rate}...")

                    if swa_active:
                        # SWA 限制：如果当前 validation loss 与最佳 loss 差距大于阈值 (默认 6%)，则跳过本次 SWA 更新
                        if (vali_loss - best_val) / best_val <= self.args.swa_loss_threshold:
                            swa_model.update_parameters(self.model)
                            swa_scheduler.step()
                        else:
                            print(f">>> Epoch {epoch+1}: Vali Loss ({vali_loss:.6f}) is > {self.args.swa_loss_threshold*100:.1f}% higher than Best ({best_val:.6f}). Skipping SWA update.")
                    else:
                        adjust_learning_rate(model_optim, epoch + 1, self.args)
                else:
                    adjust_learning_rate(model_optim, epoch + 1, self.args)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        print("average training time: {:4f} s".format(np.average(epoch_times)))
        
        # load best model: prefer in-memory if available
        # 修正：如果开启了 SWA 且已触发，则优先使用 SWA 权重；否则再使用单体最优
        if self.args.use_swa and swa_active:
            print("Applying SWA: Updating BN statistics and swapping model weights...")
            update_bn(train_loader, swa_model, device=self.device)
            self.model.load_state_dict(swa_model.module.state_dict())
        elif getattr(self.args, 'bestmodel', False) and self.best_model_state is not None:
            print("Loading best model from memory...")
            self.model.load_state_dict(self.best_model_state)
        else:
            best_model_path = path + '/' + 'checkpoint.pth'
            if os.path.exists(best_model_path):
                print(f"Loading best model from {best_model_path}...")
                self.model.load_state_dict(torch.load(best_model_path))
            else:
                if early_stopping.verbose:
                    print(f"No checkpoint found at {best_model_path}; using current model state.")
        
        print(f"Max Memory (MB): {max_memory}")
        
        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []

        self.model.in_layer.eval()
        self.model.out_layer.eval()
        self.model.time_proj.eval()
        self.model.text_proj.eval()

        with torch.no_grad(): 
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Use autocast for validation if AMP is enabled
                if self.args.use_amp:
                    with autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)
                    
                outputs_ensemble = outputs['outputs_time'] 
                outputs_ensemble = outputs_ensemble[:, -self.args.pred_len:, :]
                
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                pred = outputs_ensemble.detach().cpu()
                true = batch_y.detach().cpu()

                loss = F.mse_loss(pred, true)

                total_loss.append(loss)

        total_loss = np.average(total_loss)

        self.model.in_layer.train()
        self.model.out_layer.train()
        self.model.time_proj.train()
        self.model.text_proj.train()

        return total_loss

    def test(self, setting, test=0):
        # zero shot
        if self.args.zero_shot:
            self.args.data = self.args.target_data
            self.args.data_path = f"{self.args.data}.csv"

        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            # prefer loading best model from memory if requested and available
            if getattr(self.args, 'bestmodel', False) and getattr(self, 'best_model_state', None) is not None:
                print('Loading best model from memory...')
                self.model.load_state_dict(self.best_model_state)
            else:
                best_model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
                if os.path.exists(best_model_path):
                    self.model.load_state_dict(torch.load(best_model_path))
                else:
                    print(f"No checkpoint found at {best_model_path}; using current model state.")

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x[:, -self.args.seq_len:, :])

                outputs_ensemble = outputs['outputs_time']
                outputs_ensemble = outputs_ensemble[:, -self.args.pred_len:, :]
                
                batch_y = batch_y[:, -self.args.pred_len:, :]

                pred = outputs_ensemble.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # # # 模型参数量：
        # x = batch_x[0,:,:].unsqueeze(0)
        # test_params_flop(self.model, (x,))
        # # x = batch_x[0,:,:].unsqueeze(0).unsqueeze(0)
        # # test_params_flop(self.model, x) 
        y = (batch_x.shape[-2], batch_x.shape[-1])
        test_params_flop(self.model, y) 
        
        # preds = np.array(preds)
        # trues = np.array(trues)
        preds = np.concatenate(preds, axis=0) # without the "drop-last" trick
        trues = np.concatenate(trues, axis=0) # without the "drop-last" trick
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{},train_epoch:{}'.format(mse, mae, self.best_model_epoch))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        # ---- Save predictions CSV (前10个站点 × 间隔pred_len的10个窗口) ----
        csv_path = f"./results/feature{self.args.feature_w}_output{self.args.output_w}_mse{mse:.6f}.csv"
        def to_numpy(arr):
            if hasattr(arr, 'cpu'):
                arr = arr.detach().cpu().numpy()
            elif hasattr(arr, 'numpy'):
                arr = arr.numpy()
            return np.ascontiguousarray(arr)

        preds_np = to_numpy(preds)
        trues_np = to_numpy(trues)

        if preds_np.ndim != 3 or trues_np.ndim != 3:
            raise ValueError("preds and trues must be 3D arrays")

        num_windows, pred_len, n_features = preds_np.shape
        # 前10个站点
        n_stations = min(10, n_features)
        # 等间隔取10个窗口，起点相隔 pred_len
        n_windows_save = min(10, num_windows // pred_len)
        window_indices = [i * pred_len for i in range(n_windows_save)]

        first_chunk = True

        for idx, w in enumerate(window_indices):
            rows = []
            for s in range(pred_len):
                row = {'window': idx, 'step': s}
                for f in range(n_stations):
                    row[f'station_{f}_pred'] = preds_np[w, s, f]
                    row[f'station_{f}_true'] = trues_np[w, s, f]
                rows.append(row)

            df_chunk = pd.DataFrame(rows)
            df_chunk.to_csv(
                csv_path,
                mode='w' if first_chunk else 'a',
                header=first_chunk,
                index=False,
            )
            first_chunk = False

        print(f'Predictions saved to {csv_path} (stations=0..{n_stations-1}, windows={window_indices})')

        # ---- End CSV save ----

        return
