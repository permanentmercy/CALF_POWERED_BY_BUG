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
        param_dict = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and '_proj' in n], "lr": 1e-4},
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and '_proj' not in n], "lr": self.args.learning_rate}
        ]
        model_optim = optim.Adam([param_dict[1]], lr=self.args.learning_rate)
        loss_optim = optim.Adam([param_dict[0]], lr=self.args.learning_rate)

        return model_optim, loss_optim

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

        model_optim, loss_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        # Initialize GradScaler for AMP
        scaler = GradScaler() if self.args.use_amp else None
        if self.args.use_amp:
            print("Mixed Precision Training (AMP) Enabled")
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        
        epoch_times = []
        best_val = np.Inf
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            max_memory = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                loss_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # Forward pass with AMP if enabled
                if self.args.use_amp:
                    with autocast():
                        outputs_dict = self.model(batch_x)
                        loss = criterion(outputs_dict, batch_y)
                else:
                    outputs_dict = self.model(batch_x)
                    loss = criterion(outputs_dict, batch_y)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Backward pass with AMP if enabled
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.step(loss_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    loss_optim.step()
                
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
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))

            if self.args.cos:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        print("average training time: {:4f} s".format(np.average(epoch_times)))
        
        # load best model: prefer in-memory if available
        if getattr(self.args, 'bestmodel', False) and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        else:
            best_model_path = path + '/' + 'checkpoint.pth'
            if os.path.exists(best_model_path):
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

        return
