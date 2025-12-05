% plot_comprehensive_benchmark.m
% Visualizes results from comprehensive_benchmark.csv
% Compares OriginalKPAX, KPAX+SpatialHash, and PruneKPAX across environments

clear; close all; clc;

%% Load data
data = readtable('../Data/Benchmarks/comprehensive_benchmark.csv');

%% Extract unique planners and environments
planners = unique(data.planner, 'stable');
environments = unique(data.environment, 'stable');

num_planners = length(planners);
num_envs = length(environments);

%% Define colors for each planner
colors = [
    0.8500, 0.3250, 0.0980;  % OriginalKPAX - Orange
    0.0000, 0.4470, 0.7410;  % KPAX_SpatialHash - Blue
    0.4660, 0.6740, 0.1880;  % PruneKPAX - Green
];

%% Create figure with multiple subplots
figure('Position', [100, 100, 1400, 900]);

%% 1. Box plots for each environment
for env_idx = 1:num_envs
    subplot(2, 3, env_idx);

    % Filter data for this environment
    env_data = data(strcmp(data.environment, environments{env_idx}), :);

    % Prepare data for boxplot
    plot_data = [];
    group_labels = {};

    for p_idx = 1:num_planners
        planner_data = env_data(strcmp(env_data.planner, planners{p_idx}), :);
        times = planner_data.execution_time;
        plot_data = [plot_data; times];
        group_labels = [group_labels; repmat(planners(p_idx), length(times), 1)];
    end

    % Create boxplot
    boxplot(plot_data, group_labels, 'Colors', colors);

    title(sprintf('%s Environment', environments{env_idx}), 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Execution Time (s)', 'FontSize', 10);
    xlabel('Planner', 'FontSize', 10);
    grid on;
    set(gca, 'XTickLabelRotation', 15);
end

%% 2. Success rate comparison
subplot(2, 3, 4);

success_rates = zeros(num_envs, num_planners);
for env_idx = 1:num_envs
    for p_idx = 1:num_planners
        mask = strcmp(data.environment, environments{env_idx}) & ...
               strcmp(data.planner, planners{p_idx});
        success_rates(env_idx, p_idx) = sum(data.success(mask)) / sum(mask) * 100;
    end
end

bar_handle = bar(success_rates);
for p_idx = 1:num_planners
    bar_handle(p_idx).FaceColor = colors(p_idx, :);
end

set(gca, 'XTickLabel', environments);
ylabel('Success Rate (%)', 'FontSize', 10);
xlabel('Environment', 'FontSize', 10);
title('Success Rate by Planner', 'FontSize', 12, 'FontWeight', 'bold');
legend(planners, 'Location', 'best', 'Interpreter', 'none');
grid on;
ylim([0, 110]);

%% 3. Mean execution time comparison
subplot(2, 3, 5);

mean_times = zeros(num_envs, num_planners);
for env_idx = 1:num_envs
    for p_idx = 1:num_planners
        mask = strcmp(data.environment, environments{env_idx}) & ...
               strcmp(data.planner, planners{p_idx});
        mean_times(env_idx, p_idx) = mean(data.execution_time(mask));
    end
end

bar_handle = bar(mean_times);
for p_idx = 1:num_planners
    bar_handle(p_idx).FaceColor = colors(p_idx, :);
end

set(gca, 'XTickLabel', environments);
ylabel('Mean Execution Time (s)', 'FontSize', 10);
xlabel('Environment', 'FontSize', 10);
title('Mean Execution Time by Planner', 'FontSize', 12, 'FontWeight', 'bold');
legend(planners, 'Location', 'best', 'Interpreter', 'none');
grid on;

%% 4. Statistical summary table
subplot(2, 3, 6);
axis off;

% Create summary statistics
summary_text = cell(num_planners + 1, 5);
summary_text{1, 1} = 'Planner';
summary_text{1, 2} = 'Mean (s)';
summary_text{1, 3} = 'Median (s)';
summary_text{1, 4} = 'Std Dev (s)';
summary_text{1, 5} = 'Success %';

for p_idx = 1:num_planners
    mask = strcmp(data.planner, planners{p_idx});
    times = data.execution_time(mask);
    successes = data.success(mask);

    summary_text{p_idx + 1, 1} = planners{p_idx};
    summary_text{p_idx + 1, 2} = sprintf('%.3f', mean(times));
    summary_text{p_idx + 1, 3} = sprintf('%.3f', median(times));
    summary_text{p_idx + 1, 4} = sprintf('%.3f', std(times));
    summary_text{p_idx + 1, 5} = sprintf('%.1f%%', sum(successes)/length(successes)*100);
end

% Display as text table
y_pos = 0.9;
for row = 1:size(summary_text, 1)
    if row == 1
        weight = 'bold';
        size = 11;
    else
        weight = 'normal';
        size = 10;
    end

    text(0.05, y_pos, summary_text{row, 1}, 'FontSize', size, 'FontWeight', weight, 'Interpreter', 'none');
    text(0.35, y_pos, summary_text{row, 2}, 'FontSize', size, 'FontWeight', weight);
    text(0.50, y_pos, summary_text{row, 3}, 'FontSize', size, 'FontWeight', weight);
    text(0.65, y_pos, summary_text{row, 4}, 'FontSize', size, 'FontWeight', weight);
    text(0.82, y_pos, summary_text{row, 5}, 'FontSize', size, 'FontWeight', weight);

    y_pos = y_pos - 0.15;
end

title('Overall Statistics', 'FontSize', 12, 'FontWeight', 'bold');

%% Add main title
sgtitle('Comprehensive Planner Benchmark Results', 'FontSize', 14, 'FontWeight', 'bold');

%% Save figure
saveas(gcf, '../Data/Benchmarks/comprehensive_benchmark_plots.png');
saveas(gcf, '../Data/Benchmarks/comprehensive_benchmark_plots.fig');

fprintf('Plots saved to ../Data/Benchmarks/\n');

%% Print detailed statistics to console
fprintf('\n=== DETAILED STATISTICS ===\n\n');
for env_idx = 1:num_envs
    fprintf('--- %s Environment ---\n', environments{env_idx});
    for p_idx = 1:num_planners
        mask = strcmp(data.environment, environments{env_idx}) & ...
               strcmp(data.planner, planners{p_idx});
        times = data.execution_time(mask);
        successes = data.success(mask);

        fprintf('%s:\n', planners{p_idx});
        fprintf('  Mean: %.3f s, Median: %.3f s, Std: %.3f s\n', ...
                mean(times), median(times), std(times));
        fprintf('  Min: %.3f s, Max: %.3f s\n', min(times), max(times));
        fprintf('  Success Rate: %.1f%% (%d/%d)\n\n', ...
                sum(successes)/length(successes)*100, sum(successes), length(successes));
    end
    fprintf('\n');
end

%% Create speedup analysis
fprintf('=== SPEEDUP ANALYSIS ===\n\n');
fprintf('Speedup relative to OriginalKPAX:\n\n');

for env_idx = 1:num_envs
    fprintf('--- %s Environment ---\n', environments{env_idx});

    % Get baseline (OriginalKPAX) time
    baseline_mask = strcmp(data.environment, environments{env_idx}) & ...
                    strcmp(data.planner, 'OriginalKPAX');
    baseline_mean = mean(data.execution_time(baseline_mask));

    for p_idx = 2:num_planners  % Skip OriginalKPAX
        mask = strcmp(data.environment, environments{env_idx}) & ...
               strcmp(data.planner, planners{p_idx});
        planner_mean = mean(data.execution_time(mask));
        speedup = baseline_mean / planner_mean;

        fprintf('%s: %.2fx speedup (%.3f s vs %.3f s)\n', ...
                planners{p_idx}, speedup, baseline_mean, planner_mean);
    end
    fprintf('\n');
end

fprintf('Analysis complete!\n');
