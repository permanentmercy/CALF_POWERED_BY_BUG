%% Cross-Model External Benchmarks Plotting Script
% Models: CALF vs TQNet vs CycleNet
% Datasets: Solar-AL and Solar-IL

clear; clc; close all;

% Set directories
curr_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(curr_dir, 'data');
fig_dir = fullfile(curr_dir, 'figures');
if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

artifact_fig_dir = 'C:\Users\sheng\.gemini\antigravity\brain\d006b4c2-088d-4852-a084-081cb9ace8e4\figures';
if ~exist(artifact_fig_dir, 'dir')
    mkdir(artifact_fig_dir);
end

% Set default styling
set(0, 'DefaultAxesFontName', 'Helvetica');
set(0, 'DefaultTextFontName', 'Helvetica');
set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultAxesLineWidth', 1.2);

%% 1. Load Data
fewshot_summary = readtable(fullfile(data_dir, 'fewshot_cross_model_summary.csv'));
hp_phases = readtable(fullfile(data_dir, 'hp_tuning_phases_summary.csv'));
sweeps_all = readtable(fullfile(data_dir, 'hp_tuning_all_sweeps.csv'));

color_calf = [0.1216, 0.4667, 0.7059];    % Blue
color_tq = [0.8392, 0.1529, 0.1569];      % Red
color_cycle = [0.1725, 0.6275, 0.1725];   % Green

%% =========================================================================
%% Figure 1: Few-Shot Cross-Model MSE Curves (10% to 100%)
%% =========================================================================
f1 = figure('Position', [100, 100, 800, 520], 'Color', 'w');
hold on; grid on; box on;
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.5);

x = fewshot_summary.percent;
y_calf = fewshot_summary.CALF_MSE;
y_tq = fewshot_summary.TQNet_MSE;
y_cycle = fewshot_summary.CycleNet_MSE;

% Pure circle markers, solid/dashed/dash-dot lines, NO error bar caps
h1 = plot(x, y_calf, '-o', 'Color', color_calf, 'LineWidth', 2.5, 'MarkerSize', 8, ...
    'MarkerFaceColor', color_calf, 'DisplayName', 'CALF (Ours, Pretrained LLM)');
h2 = plot(x, y_tq, '--o', 'Color', color_tq, 'LineWidth', 2.2, 'MarkerSize', 8, ...
    'MarkerFaceColor', color_tq, 'DisplayName', 'TQNet (SOTA Time Series)');
h3 = plot(x, y_cycle, '-.o', 'Color', color_cycle, 'LineWidth', 2.2, 'MarkerSize', 8, ...
    'MarkerFaceColor', color_cycle, 'DisplayName', 'CycleNet-mlp (Periodic Baseline)');

xlabel('Training Data Ratio (%)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Test MSE (Lower is Better)', 'FontSize', 12, 'FontWeight', 'bold');
title('Cross-Model Few-Shot Benchmark on Solar-IL (10% to 100%)', 'FontSize', 13, 'FontWeight', 'bold');
set(gca, 'XTick', [10, 20, 30, 40, 50, 100], 'XTickLabel', {'10%', '20%', '30%', '40%', '50%', '100%'});
ylim([0.15, 0.39]);
legend([h1, h2, h3], 'Location', 'northeast', 'FontSize', 11);

saveas(f1, fullfile(fig_dir, 'fewshot_cross_model_mse_curves.fig'));
print(f1, fullfile(fig_dir, 'fewshot_cross_model_mse_curves_matlab.png'), '-dpng', '-r300');
print(f1, fullfile(artifact_fig_dir, 'fewshot_cross_model_mse_curves_matlab.png'), '-dpng', '-r300');
close(f1);
disp('MATLAB Fig 1 saved.');

%% =========================================================================
%% Figure 2: Few-Shot Relative Degradation Bar Chart vs 100% Baseline
%% =========================================================================
f2 = figure('Position', [100, 100, 850, 520], 'Color', 'w');
hold on; grid on; box on;
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.5);

idx_sub = fewshot_summary.percent <= 50;
deg_data = [fewshot_summary.CALF_deg_pct(idx_sub), ...
            fewshot_summary.TQNet_deg_pct(idx_sub), ...
            fewshot_summary.CycleNet_deg_pct(idx_sub)];

b2 = bar(deg_data, 'grouped');
b2(1).FaceColor = color_calf;
b2(2).FaceColor = color_tq;
b2(3).FaceColor = color_cycle;

xlabel('Training Data Ratio (%)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Degradation vs Own 100% Baseline (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('Performance Degradation Under Extreme Low-Data Regimes (vs 100% Baseline)', 'FontSize', 13, 'FontWeight', 'bold');
set(gca, 'XTick', 1:5, 'XTickLabel', {'10%', '20%', '30%', '40%', '50%'});
ylim([0, 120]);
legend({'CALF (Ours)', 'TQNet', 'CycleNet-mlp'}, 'Location', 'northeast', 'FontSize', 11);

for i = 1:3
    xpos = b2(i).XEndPoints;
    ypos = b2(i).YEndPoints;
    for j = 1:length(xpos)
        text(xpos(j), ypos(j) + 2, sprintf('+%.1f%%', deg_data(j, i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'FontSize', 8.5, 'FontWeight', 'bold');
    end
end

saveas(f2, fullfile(fig_dir, 'fewshot_cross_model_degradation_bar.fig'));
print(f2, fullfile(fig_dir, 'fewshot_cross_model_degradation_bar_matlab.png'), '-dpng', '-r300');
print(f2, fullfile(artifact_fig_dir, 'fewshot_cross_model_degradation_bar_matlab.png'), '-dpng', '-r300');
close(f2);
disp('MATLAB Fig 2 saved.');

%% =========================================================================
%% Figure 3: CALF Advantage Margin Over Baseline Models
%% =========================================================================
f3 = figure('Position', [100, 100, 850, 520], 'Color', 'w');
hold on; grid on; box on;
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.5);

adv_data = [fewshot_summary.TQNet_vs_CALF_adv_pct, fewshot_summary.CycleNet_vs_CALF_adv_pct];
b3 = bar(adv_data, 'grouped');
b3(1).FaceColor = [0.1686, 0.3608, 0.5608];
b3(2).FaceColor = [0.2627, 0.5922, 0.4588];

xlabel('Training Data Ratio (%)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('CALF Relative Advantage Margin (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('CALF Superiority Over Baselines Across Training Ratios (Solar-IL)', 'FontSize', 13, 'FontWeight', 'bold');
set(gca, 'XTick', 1:6, 'XTickLabel', {'10%', '20%', '30%', '40%', '50%', '100%'});
ylim([0, 92]);
legend({'Advantage over TQNet', 'Advantage over CycleNet-mlp'}, 'Location', 'northeast', 'FontSize', 11);

for i = 1:2
    xpos = b3(i).XEndPoints;
    ypos = b3(i).YEndPoints;
    for j = 1:length(xpos)
        text(xpos(j), ypos(j) + 1.5, sprintf('+%.1f%%', adv_data(j, i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'FontSize', 8.5, 'FontWeight', 'bold');
    end
end

saveas(f3, fullfile(fig_dir, 'fewshot_calf_advantage_bar.fig'));
print(f3, fullfile(fig_dir, 'fewshot_calf_advantage_bar_matlab.png'), '-dpng', '-r300');
print(f3, fullfile(artifact_fig_dir, 'fewshot_calf_advantage_bar_matlab.png'), '-dpng', '-r300');
close(f3);
disp('MATLAB Fig 3 saved.');

%% =========================================================================
%% Figure 4: HP Tuning Phases Baseline vs. Optimal MSE Comparison
%% =========================================================================
f4 = figure('Position', [100, 100, 950, 520], 'Color', 'w');
hold on; grid on; box on;
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.5);

hp_bar_data = [hp_phases.base_mse, hp_phases.best_mse];
b4 = bar(hp_bar_data, 'grouped');
b4(1).FaceColor = [0.5490, 0.5882, 0.7765];
b4(2).FaceColor = [0.5059, 0.0588, 0.4863];

phase_labels = {'Phase 1 (TQ-AL)', 'Phase 2 (TQ-IL)', 'Phase 3 (CycL-AL)', ...
                'Phase 4 (CycL-IL)', 'Phase 5 (CycM-AL)', 'Phase 6 (CycM-IL)'};
set(gca, 'XTick', 1:6, 'XTickLabel', phase_labels);
ylabel('Test MSE (Lower is Better)', 'FontSize', 12, 'FontWeight', 'bold');
title('Hyperparameter Tuning Optimization Across 6 Phases (Baseline vs Optimal)', 'FontSize', 13, 'FontWeight', 'bold');
ylim([0.14, 0.28]);
legend({'Baseline Config', 'Optimal Config'}, 'Location', 'northeast', 'FontSize', 11);

for j = 1:height(hp_phases)
    xpos = b4(2).XEndPoints(j);
    ypos = b4(2).YEndPoints(j);
    imp = hp_phases.improvement_pct(j);
    text(xpos, ypos + 0.003, sprintf('%.2f%%\n(%g/%d)', imp, hp_phases.best_lr(j), hp_phases.best_bs(j)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontSize', 8, 'FontWeight', 'bold', 'Color', [0.3, 0, 0.3]);
end

saveas(f4, fullfile(fig_dir, 'hp_tuning_phases_comparison.fig'));
print(f4, fullfile(fig_dir, 'hp_tuning_phases_comparison_matlab.png'), '-dpng', '-r300');
print(f4, fullfile(artifact_fig_dir, 'hp_tuning_phases_comparison_matlab.png'), '-dpng', '-r300');
close(f4);
disp('MATLAB Fig 4 saved.');

%% =========================================================================
%% Figure 5: Learning Rate Sensitivity & Dataset Distribution Shift (AL vs IL)
%% =========================================================================
f5 = figure('Position', [100, 100, 1000, 480], 'Color', 'w');

% Solar-AL Subplot
subplot(1, 2, 1);
hold on; grid on; box on; set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.5);
idx_al = (sweeps_all.bs == 64) & strcmp(sweeps_all.dataset, 'Solar-AL');
t_al = sweeps_all(idx_al, :);
grp_al = groupsummary(t_al, {'model', 'model_type', 'lr'}, 'mean', 'mse');

models_al = unique(strcat(grp_al.model, '_', grp_al.model_type));
for m = 1:length(models_al)
    sub = grp_al(strcmp(strcat(grp_al.model, '_', grp_al.model_type), models_al{m}), :);
    [lr_sorted, s_idx] = sort(sub.lr);
    mse_sorted = sub.mean_mse(s_idx);
    plot(lr_sorted, mse_sorted, '-o', 'LineWidth', 1.8, 'MarkerSize', 6, 'DisplayName', strrep(models_al{m}, '_', '-'));
end
set(gca, 'XScale', 'log');
xlabel('Learning Rate (log scale)', 'FontWeight', 'bold');
ylabel('Test MSE (bs=64)', 'FontWeight', 'bold');
title('Solar-AL Dataset (Opt lr ~ 3e-3 - 5e-3)', 'FontSize', 11, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);

% Solar-IL Subplot
subplot(1, 2, 2);
hold on; grid on; box on; set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.5);
idx_il = (sweeps_all.bs == 64) & strcmp(sweeps_all.dataset, 'Solar-IL');
t_il = sweeps_all(idx_il, :);
grp_il = groupsummary(t_il, {'model', 'model_type', 'lr'}, 'mean', 'mse');

models_il = unique(strcat(grp_il.model, '_', grp_il.model_type));
for m = 1:length(models_il)
    sub = grp_il(strcmp(strcat(grp_il.model, '_', grp_il.model_type), models_il{m}), :);
    [lr_sorted, s_idx] = sort(sub.lr);
    mse_sorted = sub.mean_mse(s_idx);
    plot(lr_sorted, mse_sorted, '-o', 'LineWidth', 1.8, 'MarkerSize', 6, 'DisplayName', strrep(models_il{m}, '_', '-'));
end
set(gca, 'XScale', 'log');
xlabel('Learning Rate (log scale)', 'FontWeight', 'bold');
ylabel('Test MSE (bs=64)', 'FontWeight', 'bold');
title('Solar-IL Dataset (Opt lr ~ 6e-4 - 1.4e-3)', 'FontSize', 11, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);

sgtitle('Learning Rate Sensitivity & Dataset Distribution Shift (bs=64)', 'FontSize', 13, 'FontWeight', 'bold');

saveas(f5, fullfile(fig_dir, 'hp_tuning_lr_sensitivity.fig'));
print(f5, fullfile(fig_dir, 'hp_tuning_lr_sensitivity_matlab.png'), '-dpng', '-r300');
print(f5, fullfile(artifact_fig_dir, 'hp_tuning_lr_sensitivity_matlab.png'), '-dpng', '-r300');
close(f5);
disp('MATLAB Fig 5 saved.');

%% =========================================================================
%% Figure 6: Master Dashboard (2x2 Panel)
%% =========================================================================
f6 = figure('Position', [100, 50, 1100, 800], 'Color', 'w');

% Subplot (a)
subplot(2, 2, 1);
hold on; grid on; box on; set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.5);
plot(x, y_calf, '-o', 'Color', color_calf, 'LineWidth', 2.2, 'MarkerSize', 7, 'MarkerFaceColor', color_calf);
plot(x, y_tq, '--o', 'Color', color_tq, 'LineWidth', 2.0, 'MarkerSize', 7, 'MarkerFaceColor', color_tq);
plot(x, y_cycle, '-.o', 'Color', color_cycle, 'LineWidth', 2.0, 'MarkerSize', 7, 'MarkerFaceColor', color_cycle);
xlabel('Training Data Ratio (%)', 'FontWeight', 'bold');
ylabel('Test MSE', 'FontWeight', 'bold');
title('(a) Few-Shot Test MSE Across Data Ratios', 'FontSize', 11, 'FontWeight', 'bold');
set(gca, 'XTick', [10, 20, 30, 40, 50, 100], 'XTickLabel', {'10%', '20%', '30%', '40%', '50%', '100%'});
ylim([0.15, 0.39]);
legend({'CALF (Ours)', 'TQNet', 'CycleNet-mlp'}, 'Location', 'northeast', 'FontSize', 9.5);

% Subplot (b)
subplot(2, 2, 2);
hold on; grid on; box on; set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.5);
b_deg = bar(deg_data, 'grouped');
b_deg(1).FaceColor = color_calf;
b_deg(2).FaceColor = color_tq;
b_deg(3).FaceColor = color_cycle;
xlabel('Training Data Ratio (%)', 'FontWeight', 'bold');
ylabel('Degradation vs 100% Baseline (%)', 'FontWeight', 'bold');
title('(b) Performance Degradation in Low-Data Regimes', 'FontSize', 11, 'FontWeight', 'bold');
set(gca, 'XTick', 1:5, 'XTickLabel', {'10%', '20%', '30%', '40%', '50%'});
ylim([0, 120]);
legend({'CALF', 'TQNet', 'CycleNet-mlp'}, 'Location', 'northeast', 'FontSize', 9.5);

% Subplot (c)
subplot(2, 2, 3);
hold on; grid on; box on; set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.5);
b_adv = bar(adv_data, 'grouped');
b_adv(1).FaceColor = [0.1686, 0.3608, 0.5608];
b_adv(2).FaceColor = [0.2627, 0.5922, 0.4588];
xlabel('Training Data Ratio (%)', 'FontWeight', 'bold');
ylabel('CALF Advantage Margin (%)', 'FontWeight', 'bold');
title('(c) CALF Superiority Across Training Ratios', 'FontSize', 11, 'FontWeight', 'bold');
set(gca, 'XTick', 1:6, 'XTickLabel', {'10%', '20%', '30%', '40%', '50%', '100%'});
ylim([0, 92]);
legend({'Advantage over TQNet', 'Advantage over CycleNet-mlp'}, 'Location', 'northeast', 'FontSize', 9.5);

% Subplot (d)
subplot(2, 2, 4);
hold on; grid on; box on; set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.5);
b_hp = bar(hp_bar_data, 'grouped');
b_hp(1).FaceColor = [0.5490, 0.5882, 0.7765];
b_hp(2).FaceColor = [0.5059, 0.0588, 0.4863];
ylabel('Test MSE', 'FontWeight', 'bold');
title('(d) HP Tuning Across 6 Phases', 'FontSize', 11, 'FontWeight', 'bold');
set(gca, 'XTick', 1:6, 'XTickLabel', {'P1:TQ-AL', 'P2:TQ-IL', 'P3:CycL-AL', 'P4:CycL-IL', 'P5:CycM-AL', 'P6:CycM-IL'}, 'FontSize', 8.5);
ylim([0.14, 0.28]);
legend({'Baseline', 'Optimal'}, 'Location', 'northeast', 'FontSize', 9.5);

sgtitle('External Benchmarking Master Dashboard: CALF vs TQNet & CycleNet', 'FontSize', 14, 'FontWeight', 'bold');

saveas(f6, fullfile(fig_dir, 'external_benchmarks_master_dashboard.fig'));
print(f6, fullfile(fig_dir, 'external_benchmarks_master_dashboard_matlab.png'), '-dpng', '-r300');
print(f6, fullfile(artifact_fig_dir, 'external_benchmarks_master_dashboard_matlab.png'), '-dpng', '-r300');
close(f6);
disp('MATLAB Fig 6 saved.');

disp('All MATLAB figures generated successfully!');
