import numpy as np

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs(100 * (pred - true) / (true + 1e-8)))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / (true + 1e-8)))

def metric(pred, true):
    """
    Memory-efficient metric calculation using weighted chunking.
    This prevents ArrayMemoryError on large datasets and ensures mathematical precision.
    """
    batch_size = 1000  # 每次处理1000个样本
    total_samples = pred.shape[0]
    
    mae_list, mse_list, mape_list, mspe_list = [], [], [], []
    weights = []
    
    for i in range(0, total_samples, batch_size):
        end = min(i + batch_size, total_samples)
        p_chunk = pred[i:end]
        t_chunk = true[i:end]
        chunk_weight = end - i
        
        mae_list.append(np.mean(np.abs(p_chunk - t_chunk)))
        mse_list.append(np.mean((p_chunk - t_chunk) ** 2))
        mape_list.append(np.mean(np.abs(100 * (p_chunk - t_chunk) / (t_chunk + 1e-8))))
        mspe_list.append(np.mean(np.square((p_chunk - t_chunk) / (t_chunk + 1e-8))))
        weights.append(chunk_weight)
    
    # 使用加权平均，确保结果与全量计算完全一致
    mae = np.average(mae_list, weights=weights)
    mse = np.average(mse_list, weights=weights)
    rmse = np.sqrt(mse)
    mape = np.average(mape_list, weights=weights)
    mspe = np.average(mspe_list, weights=weights)

    return mae, mse, rmse, mape, mspe
