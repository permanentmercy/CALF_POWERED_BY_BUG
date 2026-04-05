import re
import pandas as pd
import os

# ================= 配置区域 =================
INPUT_FILE = 'result_long_term_forecast.txt'  # 输入日志文件
OUTPUT_FILE = 'results.xlsx'                  # 输出 Excel 文件
# ===========================================

def parse_log_line(config_line, result_line):
    data = {}
    
    # 1. 初始化所有列 (新增 best_epoch)
    base_params = [
        'task_name', 'model_id', 'model', 'data', 
        'features', 'seq_len', 'label_len', 'pred_len', 
        'd_model', 'n_heads', 'e_layers', 'd_layers', 
        'd_ff', 'factor', 'embed', 'distil', 'des', 
        'gpt_layers', 'learning_rate', 'random_seed', 'ii', 
        'feature_w', 'output_w', 'train_epochs',
        'batch_size', 'dropout', 'lora_dropout', 'lora_r',
        'mse', 'mae', 'best_epoch'  # 新增
    ]
    for key in base_params:
        data[key] = ""

    # 2. 定义正则表达式模式 (新增 train_epoch 匹配)
    patterns = {
        'features': r'ft([A-Za-z0-9]+)',
        'seq_len': r'sl(\d+)',
        'label_len': r'll(\d+)',
        'pred_len': r'pl(\d+)',
        'd_model': r'dm(\d+)',
        'n_heads': r'nh(\d+)',
        'e_layers': r'el(\d+)',
        'd_layers': r'dl(\d+)',
        'd_ff': r'df(\d+)',
        'factor': r'fc(\d+)',
        'embed': r'eb(.*?)_dt',
        'distil': r'dt(True|False)',
        'gpt_layers': r'gpt(\d+)',
        'learning_rate': r'rl([\d.]+)',
        'feature_w': r'feature([\d.]+)',
        'output_w': r'output([\d.]+)',
        'train_epochs': r'train_epochs(\d+)',
        'batch_size': r'bs(\d+)',
        'dropout': r'dr([\d.]+)',
        'lora_dropout': r'ld([\d.]+)',
        'lora_r': r'r(\d+)',
        'mse': r'mse:([\d.]+)',
        'mae': r'mae:([\d.]+)',
        'best_epoch': r'train_epoch:(\d+)'  # 新增：匹配 train_epoch:数字
    }

    full_text = f"{config_line} {result_line}"

    for key, pattern in patterns.items():
        match = re.search(pattern, full_text)
        if match:
            data[key] = match.group(1)

    # 3. 特殊位置参数提取
    des_match = re.search(r'dt(?:True|False)_(.*?)_gpt', config_line)
    if des_match:
        data['des'] = des_match.group(1)

    seed_ii_match = re.search(r'rl[\d.]+_(\d+)_(\d+)_feature', config_line)
    if seed_ii_match:
        data['random_seed'] = seed_ii_match.group(1)
        data['ii'] = seed_ii_match.group(2)

    prefix_match = re.search(r'^(.*?)_ft', config_line)
    if prefix_match:
        prefix_str = prefix_match.group(1)
        parts = prefix_str.split('_')
        if len(parts) >= 4:
            data['data'] = parts[-1]
            data['model'] = parts[-2]
            data['model_id'] = parts[-3]
            data['task_name'] = "_".join(parts[:-3])
        else:
            data['task_name'] = prefix_str

    return data

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"未找到 {INPUT_FILE}，正在使用示例数据生成演示...")
        # 示例数据包含新的 train_epoch 字段
        sample_data = """
long_term_forecast_CALF_96_96_CALF_Solar_ftM_sl96_ll0_pl96_dm768_nh4_el2_dl1_df384_fc1_ebtimeF_dtTrue_test_gpt3_rl0.0001_2026_0_feature0.018_output1.5_train_epochs3_bs32_dr0.3_ld0.1_r8_
mse:0.2276013046503067, mae:0.26368096470832825, train_epoch:2

long_term_forecast_CALF_96_96_CALF_Solar_ftM_sl96_ll0_pl96_dm768_nh4_el2_dl1_df384_fc1_ebtimeF_dtTrue_test_gpt3_rl0.0001_2026_0_feature0.3_output1.1_train_epochs3_bs32_dr0.3_ld0.1_r16_
mse:0.23402352631092072, mae:0.27033278346061707, train_epoch:3
"""
        with open(INPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(sample_data)

    results = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines if line.strip()]

    i = 0
    while i < len(lines):
        config_line = lines[i]
        result_line = ""
        
        # 检查下一行是否包含结果信息 (mse/mae)
        if i + 1 < len(lines) and ('mse' in lines[i+1] or 'mae' in lines[i+1]):
            result_line = lines[i+1]
            i += 2
        else:
            i += 1
            
        parsed_data = parse_log_line(config_line, result_line)
        results.append(parsed_data)

    if results:
        df = pd.DataFrame(results)
        
        # 优化列顺序 (将 best_epoch 放在 mse/mae 旁边)
        priority_cols = [
            'task_name', 'model', 'data', 'seq_len', 'pred_len', 
            'lora_r', 'lora_dropout', 'dropout', 'batch_size',
            'learning_rate', 'train_epochs', 'best_epoch',  # best_epoch 靠近 train_epochs
            'feature_w', 'output_w', 
            'mse', 'mae'
        ]
        existing_cols = df.columns.tolist()
        ordered_cols = [c for c in priority_cols if c in existing_cols] + \
                       [c for c in existing_cols if c not in priority_cols]
        
        df = df[ordered_cols]

        # ================= 写入 Excel 并设置格式 =================
        try:
            writer = pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter')
            df.to_excel(writer, index=False, sheet_name='Results')
            
            workbook = writer.book
            worksheet = writer.sheets['Results']
            
            # 定义格式
            header_format = workbook.add_format({
                'font_name': 'SimHei',
                'bold': True,
                'align': 'center',
                'valign': 'vcenter',
                'bg_color': '#D7E4BD',
                'border': 1
            })
            
            cell_format = workbook.add_format({
                'font_name': 'SimHei',
                'align': 'center',
                'valign': 'vcenter',
                'border': 1
            })

            # 应用表头格式
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # 应用单元格格式
            for row in range(1, len(df) + 1):
                for col in range(len(df.columns)):
                    value = df.iloc[row-1, col]
                    worksheet.write(row, col, value, cell_format)

            # 自动调整列宽
            for i, col in enumerate(df.columns):
                column_len = max(len(str(col)), df[col].astype(str).map(len).max())
                width = min(max(column_len + 2, 10), 50) 
                worksheet.set_column(i, i, width)
            
            writer.close()
            print(f"✅ 成功提取 {len(results)} 条记录，已保存至 {OUTPUT_FILE}")
            print(f"📋 包含列：{', '.join(df.columns)}")
            
        except Exception as e:
            print(f"❌ 写入 Excel 时出错：{e}")
            print("💡 请确保安装了 xlsxwriter: pip install xlsxwriter")
            df.to_excel(OUTPUT_FILE, index=False)

    else:
        print("❌ 未找到任何有效的日志记录。")

if __name__ == '__main__':
    main()