%% =========================================================================
% CALF: 学习率敏感性与预测步长分析绘图脚本 (MATLAB)
% 
% 说明：
%   1. 读取 84 组实验结果（4 预测步长 x 7 学习率 x 3 随机种子）；
%   2. 对每个 (pred_len, learning_rate) 组合计算 3 个随机种子的均值；
%   3. 在同一张图中绘制 4 条不同预测步长（Horizon 96, 192, 336, 720）随学习率变化的折线图；
%   4. 输出学术出版级 300 DPI PNG、矢量 PDF 以及 MATLAB 交互式 FIG 文件。
% =========================================================================

function plot_lr_vs_mse()
    clear; clc; close all;

    % 1. 获取当前脚本所在目录
    script_dir = fileparts(mfilename('fullpath'));
    if isempty(script_dir)
        script_dir = pwd;
    end
    cd(script_dir);
    fprintf('=== [CALF] 学习率敏感性折线图绘制 ===\n');
    fprintf('工作目录: %s\n', script_dir);

    % 2. 数据读取 (三级冗余容错机制)
    csv_file = fullfile(script_dir, 'learning_rate_summary.csv');
    json_file = fullfile(script_dir, 'batch_state.json');

    pred_lens = [96, 192, 336, 720];
    lrs = [0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.00045, 0.0005];
    n_h = length(pred_lens);
    n_lr = length(lrs);

    mse_matrix = zeros(n_h, n_lr);
    std_matrix = zeros(n_h, n_lr);

    loaded = false;

    % 途径 A: 读取汇总 CSV
    if exist(csv_file, 'file')
        try
            opts = detectImportOptions(csv_file);
            t = readtable(csv_file, opts);
            for i = 1:n_h
                pl = pred_lens(i);
                for j = 1:n_lr
                    lr_val = lrs(j);
                    row = t(t.pred_len == pl & abs(t.learning_rate - lr_val) < 1e-7, :);
                    if ~isempty(row)
                        mse_matrix(i, j) = row.mse_mean(1);
                        if ismember('mse_std', row.Properties.VariableNames)
                            std_matrix(i, j) = row.mse_std(1);
                        end
                    end
                end
            end
            loaded = true;
            fprintf('[INFO] 成功从 CSV 文件读取汇总数据: %s\n', csv_file);
        catch ME
            fprintf('[WARN] CSV 读取失败 (%s)，尝试解析 JSON...\n', ME.message);
        end
    end

    % 途径 B: 读取 batch_state.json
    if ~loaded && exist(json_file, 'file')
        try
            raw = fileread(json_file);
            data = jsondecode(raw);
            tasks = struct2cell(data.tasks);
            
            % 收集所有已完成任务的数据
            temp_records = [];
            for k = 1:length(tasks)
                t_k = tasks{k};
                if strcmp(t_k.status, 'COMPLETED')
                    temp_records = [temp_records; t_k.loop_signature.pred_len, ...
                                                  t_k.loop_signature.learning_rate, ...
                                                  t_k.loop_signature.random_seed, ...
                                                  t_k.metrics.mse];
                end
            end
            
            for i = 1:n_h
                pl = pred_lens(i);
                for j = 1:n_lr
                    lr_val = lrs(j);
                    matches = temp_records(temp_records(:,1) == pl & abs(temp_records(:,2) - lr_val) < 1e-7, 4);
                    if ~isempty(matches)
                        mse_matrix(i, j) = mean(matches);
                        std_matrix(i, j) = std(matches);
                    end
                end
            end
            loaded = true;
            fprintf('[INFO] 成功从 batch_state.json 解析并聚合 3 种子均值\n');
        catch ME
            fprintf('[WARN] JSON 读取失败 (%s)，使用内置数据矩阵...\n', ME.message);
        end
    end

    % 途径 C: 内置完整数据备份 (保障离线无依赖独立运行)
    if ~loaded
        fprintf('[INFO] 使用 84 组实验精确测量基准数据\n');
        mse_matrix = [
            0.174846, 0.172591, 0.169271, 0.166955, 0.164683, 0.168275, 0.168708;  % Pred 96
            0.200705, 0.197043, 0.192129, 0.190781, 0.190280, 0.190103, 0.190562;  % Pred 192
            0.201237, 0.208890, 0.205849, 0.201390, 0.199336, 0.199445, 0.200798;  % Pred 336
            0.207627, 0.208639, 0.207640, 0.205690, 0.206460, 0.206701, 0.206205   % Pred 720
        ];
        std_matrix = [
            0.004248, 0.000314, 0.001424, 0.001101, 0.000475, 0.003693, 0.003191;
            0.006801, 0.002331, 0.002543, 0.001265, 0.003179, 0.003303, 0.001372;
            0.002137, 0.004699, 0.000756, 0.000978, 0.000834, 0.000776, 0.002975;
            0.001321, 0.002163, 0.001288, 0.001391, 0.001022, 0.001208, 0.001201
        ];
    end

    % 3. 打印数据预览
    fprintf('\n---------------- 3 种子均值 MSE 统计表 ----------------\n');
    fprintf('%-10s', 'PredLen');
    for j = 1:n_lr
        fprintf('%-11.5f', lrs(j));
    end
    fprintf('\n');
    for i = 1:n_h
        fprintf('%-10d', pred_lens(i));
        for j = 1:n_lr
            fprintf('%-11.6f', mse_matrix(i, j));
        end
        fprintf('\n');
    end
    fprintf('-------------------------------------------------------\n\n');

    % 4. 绘图配置与风格美化 (IEEE / Nature 学术论文规范)
    fig = figure('Name', 'CALF Learning Rate Sensitivity', ...
                 'Color', 'w', ...
                 'Units', 'pixels', ...
                 'Position', [150, 150, 960, 620], ...
                 'Visible', 'on');
    hold on;

    % 配色方案: 经典学术四色 (区分明显，黑白打印亦可辨认)
    colors = [
        0.1216, 0.4667, 0.7059;   % Horizon 96:  经典蓝 (Deep Blue)
        0.8500, 0.3725, 0.0078;   % Horizon 192: 珊瑚橙 (Vibrant Orange)
        0.1725, 0.6275, 0.1725;   % Horizon 336: 翡翠绿 (Forest Green)
        0.8392, 0.1529, 0.1569    % Horizon 720: 宝石红 (Crimson Red)
    ];

    markers = {'o', 's', '^', 'd'};
    line_styles = {'-', '-', '-', '-'};

    plot_handles = gobjects(n_h, 1);

    % 绘制折线图
    for i = 1:n_h
        plot_handles(i) = plot(lrs, mse_matrix(i, :), ...
            'LineStyle', line_styles{i}, ...
            'Color', colors(i, :), ...
            'LineWidth', 2.2, ...
            'Marker', markers{i}, ...
            'MarkerSize', 8, ...
            'MarkerEdgeColor', colors(i, :) * 0.8, ...
            'MarkerFaceColor', colors(i, :), ...
            'DisplayName', sprintf('Pred Len = %d', pred_lens(i)));
    end

    % 标注默认/最优学习率 0.00040 参考基准线
    xline_h = xline(0.0004, '--', 'Color', [0.4, 0.4, 0.4], 'LineWidth', 1.4, ...
                    'DisplayName', 'Default lr = 0.00040');

    % 坐标轴与范围设置
    xlim([0.00018, 0.00052]);
    xticks(lrs);
    xticklabels({'0.00020', '0.00025', '0.00030', '0.00035', '0.00040', '0.00045', '0.00050'});

    % 优化 Y 轴显示范围 (留出足够顶部与底部呼吸空间)
    ylim([0.160, 0.215]);
    yticks(0.160:0.010:0.215);

    % 轴标签与标题
    xlabel('Learning Rate (\eta)', 'FontSize', 13, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
    ylabel('Mean Squared Error (MSE)', 'FontSize', 13, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
    title('CALF: Sensitivity of Forecasting MSE to Learning Rate (Averaged over 3 Seeds)', ...
          'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Times New Roman');

    % 图例设置：移至顶部图外水平居中展示，彻底杜绝遮挡任何折线
    leg = legend([plot_handles; xline_h], ...
                 'Location', 'northoutside', ...
                 'Orientation', 'horizontal', ...
                 'FontSize', 11, ...
                 'FontName', 'Times New Roman', ...
                 'Box', 'off');

    % 网格与精细刻度
    grid on;
    box on;
    set(gca, 'FontSize', 12, ...
             'FontName', 'Times New Roman', ...
             'LineWidth', 1.2, ...
             'GridLineStyle', ':', ...
             'GridAlpha', 0.4, ...
             'MinorGridAlpha', 0.2, ...
             'TickDir', 'in', ...
             'TickLength', [0.015, 0.015]);

    hold off;

    % 5. 导出保存图 1 产物至当前目录 (checkpoints/学习率/)
    png_path = fullfile(script_dir, 'learning_rate_vs_mse.png');
    pdf_path = fullfile(script_dir, 'learning_rate_vs_mse.pdf');
    fig_path = fullfile(script_dir, 'learning_rate_vs_mse.fig');

    try
        exportgraphics(fig, png_path, 'Resolution', 300);
        fprintf('[OK] 图1 高清图片已保存: %s (300 DPI)\n', png_path);
    catch
        print(fig, png_path, '-dpng', '-r300');
        fprintf('[OK] 图1 高清图片已保存 (print): %s\n', png_path);
    end

    try
        exportgraphics(fig, pdf_path, 'ContentType', 'vector');
        fprintf('[OK] 图1 矢量 PDF 已保存: %s\n', pdf_path);
    catch
    end
    savefig(fig, fig_path);
    fprintf('[OK] 图1 MATLAB 工程已保存: %s\n', fig_path);

    % =====================================================================
    % 6. 图 2: 四个预测步长取平均的整体折线图 (Overall 4-Horizon Average)
    % =====================================================================
    fig2 = figure('Name', 'CALF 4-Horizon Average Learning Rate Sensitivity', ...
                  'Color', 'w', ...
                  'Units', 'pixels', ...
                  'Position', [180, 180, 920, 600], ...
                  'Visible', 'on');
    hold on;

    avg_mse = mean(mse_matrix, 1);    % 沿步长维度取平均 (1 x 7)
    std_mse = std(mse_matrix, 0, 1);   % 4 个步长之间的离散标准差

    % 打印 4 步长平均统计
    fprintf('\n---------------- 4 步长综合平均 MSE ----------------\n');
    for j = 1:n_lr
        fprintf('  lr = %.5f: Mean = %.6f, Std = %.6f\n', lrs(j), avg_mse(j), std_mse(j));
    end
    [min_val, min_idx] = min(avg_mse);
    min_lr = lrs(min_idx);
    fprintf('  ==> 全局最优学习率: %.5f (4 步长综合 MSE = %.6f)\n', min_lr, min_val);
    fprintf('-----------------------------------------------------\n\n');

    % 绘制标准差轻量半透明阴影带 (展现各步长离散程度)
    fill_x = [lrs, fliplr(lrs)];
    fill_y = [avg_mse - std_mse, fliplr(avg_mse + std_mse)];
    h_fill = fill(fill_x, fill_y, [0.15, 0.45, 0.80], ...
                  'FaceAlpha', 0.12, 'EdgeColor', 'none', ...
                  'DisplayName', '\pm1 Std Dev across Horizons');

    % 绘制主折线 (藏青色 + 实心圆点)
    main_color = [0.12, 0.38, 0.72];
    h_line = plot(lrs, avg_mse, '-o', ...
                  'Color', main_color, ...
                  'LineWidth', 2.5, ...
                  'MarkerSize', 8.5, ...
                  'MarkerFaceColor', main_color, ...
                  'MarkerEdgeColor', [0.08, 0.22, 0.45], ...
                  'DisplayName', '4-Horizon Mean MSE');

    % 突出标注最优最低点 (五角星标记)
    h_min = plot(min_lr, min_val, 'p', ...
                 'MarkerSize', 14, ...
                 'MarkerFaceColor', [0.85, 0.15, 0.15], ...
                 'MarkerEdgeColor', [0.55, 0.05, 0.05], ...
                 'LineWidth', 1.5, ...
                 'DisplayName', sprintf('Optimal \\eta = %.5f (MSE = %.4f)', min_lr, min_val));

    % 辅助垂直虚线标出最优学习率位置
    xline_h2 = xline(min_lr, '--', 'Color', [0.5, 0.5, 0.5], 'LineWidth', 1.2, ...
                     'HandleVisibility', 'off');

    % 在各数据点处添加精细数值标签
    for j = 1:n_lr
        if j == min_idx
            text(lrs(j), avg_mse(j) - 0.0018, sprintf('Optimal: %.4f', avg_mse(j)), ...
                 'FontSize', 10.5, 'FontWeight', 'bold', 'FontName', 'Times New Roman', ...
                 'HorizontalAlignment', 'center', 'Color', [0.85, 0.15, 0.15]);
        else
            text(lrs(j), avg_mse(j) + 0.0012, sprintf('%.4f', avg_mse(j)), ...
                 'FontSize', 10, 'FontName', 'Times New Roman', ...
                 'HorizontalAlignment', 'center', 'Color', [0.25, 0.25, 0.25]);
        end
    end

    xlim([0.00018, 0.00052]);
    xticks(lrs);
    xticklabels({'0.00020', '0.00025', '0.00030', '0.00035', '0.00040', '0.00045', '0.00050'});
    ylim([0.170, 0.216]);
    yticks(0.170:0.010:0.210);

    xlabel('Learning Rate (\eta)', 'FontSize', 13, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
    ylabel('Average Mean Squared Error (MSE)', 'FontSize', 13, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
    title('CALF: Overall Average MSE across 4 Horizons (96, 192, 336, 720)', ...
          'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Times New Roman');

    leg2 = legend([h_line, h_min, h_fill], ...
                  'Location', 'northoutside', ...
                  'Orientation', 'horizontal', ...
                  'FontSize', 11, ...
                  'FontName', 'Times New Roman', ...
                  'Box', 'off');

    grid on;
    box on;
    set(gca, 'FontSize', 12, ...
             'FontName', 'Times New Roman', ...
             'LineWidth', 1.2, ...
             'GridLineStyle', ':', ...
             'GridAlpha', 0.4, ...
             'MinorGridAlpha', 0.2, ...
             'TickDir', 'in', ...
             'TickLength', [0.015, 0.015]);

    hold off;

    % 导出保存图 2 产物
    png_path_avg = fullfile(script_dir, 'learning_rate_vs_mse_avg.png');
    pdf_path_avg = fullfile(script_dir, 'learning_rate_vs_mse_avg.pdf');
    fig_path_avg = fullfile(script_dir, 'learning_rate_vs_mse_avg.fig');

    try
        exportgraphics(fig2, png_path_avg, 'Resolution', 300);
        fprintf('[OK] 图2 (4步长平均) 高清图片已保存: %s (300 DPI)\n', png_path_avg);
    catch
        print(fig2, png_path_avg, '-dpng', '-r300');
        fprintf('[OK] 图2 高清图片已保存 (print): %s\n', png_path_avg);
    end

    try
        exportgraphics(fig2, pdf_path_avg, 'ContentType', 'vector');
        fprintf('[OK] 图2 矢量 PDF 已保存: %s\n', pdf_path_avg);
    catch
    end
    savefig(fig2, fig_path_avg);
    fprintf('[OK] 图2 MATLAB 工程已保存: %s\n', fig_path_avg);

    fprintf('\n=== 绘图全部顺利完成！两张高清学术插图均已就绪。===\n');
end
