% PLOT_OKPAX_BENCHMARK - Visualize OKPAX comprehensive benchmark results
%
% This script loads CSV files from the OKPAX benchmark and generates:
%   1. Cost convergence plots for each environment
%   2. Tree size growth analysis
%   3. Iteration timing analysis
%   4. Summary statistics comparisons
%
% USAGE:
%   1. Download the Data/Benchmarks/OKPAX/ folder from your Linux machine
%   2. Place it in the same directory as this script, or modify DATA_DIR below
%   3. Run this script in MATLAB
%
% The script will automatically find all CSV files and generate plots.

clear; close all; clc;

%% Configuration
% Modify this path to point to your downloaded benchmark data
DATA_DIR = 'Data/Benchmarks/OKPAX';

% Create output directory for plots
OUTPUT_DIR = fullfile(DATA_DIR, 'plots');
if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end

fprintf('OKPAX Benchmark Visualization\n');
fprintf('==============================\n\n');

%% Load Data
fprintf('Loading benchmark data from: %s\n', DATA_DIR);

% Find summary file
summary_files = dir(fullfile(DATA_DIR, '*_summary.csv'));
if isempty(summary_files)
    error('No summary CSV file found in %s', DATA_DIR);
end

% Load summary data
summary_file = fullfile(DATA_DIR, summary_files(1).name);
summary_data = readtable(summary_file);
fprintf('✓ Loaded summary: %d runs\n', height(summary_data));

% Find detailed iteration files
detail_files = dir(fullfile(DATA_DIR, '*_run*.csv'));
fprintf('✓ Found %d detailed iteration files\n', length(detail_files));

% Load all detailed data
all_detailed_data = [];
for i = 1:length(detail_files)
    filename = detail_files(i).name;
    filepath = fullfile(DATA_DIR, filename);

    % Parse metadata from filename
    % Format: {timestamp}_{environment}_{planner_type}_run{N}.csv
    parts = split(filename, '_');

    % Extract environment, planner type, and run number
    % Assumes format: okpax_benchmark_{timestamp}_{environment}_{planner}_run{N}.csv
    if length(parts) >= 4
        environment = parts{end-2};
        planner_type = parts{end-1};
        run_str = parts{end};
        run_number = str2double(replace(run_str, {'.csv', 'run'}, ''));

        % Load CSV
        data = readtable(filepath);

        % Add metadata columns
        data.environment = repmat({environment}, height(data), 1);
        data.planner_type = repmat({planner_type}, height(data), 1);
        data.run_number = repmat(run_number, height(data), 1);

        all_detailed_data = [all_detailed_data; data];
    end
end

fprintf('✓ Loaded %d total iteration records\n\n', height(all_detailed_data));

% Get unique values
environments = unique(all_detailed_data.environment);
planner_types = unique(all_detailed_data.planner_type);

fprintf('Environments: %s\n', strjoin(environments, ', '));
fprintf('Planner types: %s\n\n', strjoin(planner_types, ', '));

%% Generate Plots for Each Environment
fprintf('Generating plots...\n\n');

for env_idx = 1:length(environments)
    env = environments{env_idx};
    fprintf('Processing environment: %s\n', env);

    % Filter data for this environment
    env_mask = strcmp(all_detailed_data.environment, env);
    env_data = all_detailed_data(env_mask, :);

    % Create figure with subplots
    fig = figure('Position', [100, 100, 1600, 1200]);
    sgtitle(sprintf('OKPAX Performance Analysis - %s Environment', env), ...
            'FontSize', 16, 'FontWeight', 'bold');

    %% Subplot 1: Cost Convergence (All Runs + Mean)
    subplot(2, 2, 1);
    hold on; grid on;

    colors = lines(length(planner_types));

    for p_idx = 1:length(planner_types)
        planner = planner_types{p_idx};
        planner_mask = strcmp(env_data.planner_type, planner);
        planner_data = env_data(planner_mask, :);

        % Plot individual runs (transparent)
        runs = unique(planner_data.run_number);
        for r_idx = 1:length(runs)
            run = runs(r_idx);
            run_mask = planner_data.run_number == run;
            run_data = planner_data(run_mask, :);

            plot(run_data.iteration, run_data.min_cost, ...
                 'Color', [colors(p_idx, :), 0.3], 'LineWidth', 1);
        end

        % Plot mean (bold)
        [G, iterations] = findgroups(planner_data.iteration);
        mean_cost = splitapply(@mean, planner_data.min_cost, G);

        plot(iterations, mean_cost, 'Color', colors(p_idx, :), ...
             'LineWidth', 3, 'DisplayName', planner);
    end

    xlabel('Iteration');
    ylabel('Minimum Cost');
    title('Cost Convergence (all runs + mean)');
    legend('Location', 'best');
    hold off;

    %% Subplot 2: Mean Cost with Confidence Intervals
    subplot(2, 2, 2);
    hold on; grid on;

    for p_idx = 1:length(planner_types)
        planner = planner_types{p_idx};
        planner_mask = strcmp(env_data.planner_type, planner);
        planner_data = env_data(planner_mask, :);

        % Compute mean and std by iteration
        [G, iterations] = findgroups(planner_data.iteration);
        mean_cost = splitapply(@mean, planner_data.min_cost, G);
        std_cost = splitapply(@std, planner_data.min_cost, G);

        % Plot mean line
        plot(iterations, mean_cost, 'Color', colors(p_idx, :), ...
             'LineWidth', 2, 'DisplayName', planner);

        % Plot confidence interval
        fill([iterations; flipud(iterations)], ...
             [mean_cost - std_cost; flipud(mean_cost + std_cost)], ...
             colors(p_idx, :), 'FaceAlpha', 0.2, 'EdgeColor', 'none', ...
             'HandleVisibility', 'off');
    end

    xlabel('Iteration');
    ylabel('Minimum Cost');
    title('Mean Cost ± Std Dev');
    legend('Location', 'best');
    hold off;

    %% Subplot 3: Tree Size Growth
    subplot(2, 2, 3);
    hold on; grid on;

    for p_idx = 1:length(planner_types)
        planner = planner_types{p_idx};
        planner_mask = strcmp(env_data.planner_type, planner);
        planner_data = env_data(planner_mask, :);

        % Compute mean tree size by iteration
        [G, iterations] = findgroups(planner_data.iteration);
        mean_tree_size = splitapply(@mean, planner_data.tree_size, G);

        plot(iterations, mean_tree_size, 'Color', colors(p_idx, :), ...
             'LineWidth', 2, 'DisplayName', planner);
    end

    xlabel('Iteration');
    ylabel('Tree Size');
    title('Mean Tree Size Growth');
    legend('Location', 'best');
    hold off;

    %% Subplot 4: Iteration Execution Time
    subplot(2, 2, 4);
    hold on; grid on;

    for p_idx = 1:length(planner_types)
        planner = planner_types{p_idx};
        planner_mask = strcmp(env_data.planner_type, planner);
        planner_data = env_data(planner_mask, :);

        % Compute mean execution time by iteration
        [G, iterations] = findgroups(planner_data.iteration);
        mean_time = splitapply(@mean, planner_data.execution_time_ms, G);

        plot(iterations, mean_time, 'Color', colors(p_idx, :), ...
             'LineWidth', 2, 'DisplayName', planner);
    end

    xlabel('Iteration');
    ylabel('Time (ms)');
    title('Mean Iteration Execution Time');
    legend('Location', 'best');
    hold off;

    % Save figure
    output_file = fullfile(OUTPUT_DIR, sprintf('cost_convergence_%s.png', env));
    saveas(fig, output_file);
    fprintf('  ✓ Saved: %s\n', output_file);

    % Also save as .fig for interactive viewing
    saveas(fig, fullfile(OUTPUT_DIR, sprintf('cost_convergence_%s.fig', env)));
end

%% Summary Statistics Plots
fprintf('\nGenerating summary statistics plots...\n');

fig = figure('Position', [100, 100, 1600, 1200]);
sgtitle('OKPAX Summary Statistics', 'FontSize', 16, 'FontWeight', 'bold');

% Create grouping variable for box plots
summary_data.group = strcat(summary_data.environment, '_', summary_data.planner_type);

%% Subplot 1: Final Cost Distribution
subplot(2, 2, 1);
boxplot(summary_data.final_cost, summary_data.group);
ylabel('Final Cost');
title('Final Cost Distribution');
xtickangle(45);
grid on;

%% Subplot 2: Total Time Comparison
subplot(2, 2, 2);
boxplot(summary_data.total_time_seconds, summary_data.group);
ylabel('Time (seconds)');
title('Total Execution Time');
xtickangle(45);
grid on;

%% Subplot 3: Final Iterations
subplot(2, 2, 3);
boxplot(summary_data.final_iterations, summary_data.group);
ylabel('Iterations');
title('Iterations to Convergence');
xtickangle(45);
grid on;

%% Subplot 4: Final Tree Size
subplot(2, 2, 4);
boxplot(summary_data.final_tree_size, summary_data.group);
ylabel('Tree Size');
title('Final Tree Size');
xtickangle(45);
grid on;

% Save summary figure
output_file = fullfile(OUTPUT_DIR, 'summary_statistics.png');
saveas(fig, output_file);
fprintf('✓ Saved: %s\n', output_file);
saveas(fig, fullfile(OUTPUT_DIR, 'summary_statistics.fig'));

%% Generate Text Report
fprintf('\nGenerating text report...\n');

report_file = fullfile(OUTPUT_DIR, 'benchmark_report.txt');
fid = fopen(report_file, 'w');

fprintf(fid, 'OKPAX Comprehensive Benchmark Report\n');
fprintf(fid, '=====================================\n\n');

for env_idx = 1:length(environments)
    env = environments{env_idx};
    fprintf(fid, 'Environment: %s\n', env);
    fprintf(fid, '%s\n\n', repmat('-', 1, 50));

    env_mask = strcmp(summary_data.environment, env);
    env_summary = summary_data(env_mask, :);

    for p_idx = 1:length(planner_types)
        planner = planner_types{p_idx};
        planner_mask = strcmp(env_summary.planner_type, planner);
        planner_summary = env_summary(planner_mask, :);

        if isempty(planner_summary)
            continue;
        end

        fprintf(fid, '  %s:\n', planner);
        fprintf(fid, '    Runs: %d\n', height(planner_summary));
        fprintf(fid, '    Mean Final Cost: %.3f ± %.3f\n', ...
                mean(planner_summary.final_cost), std(planner_summary.final_cost));
        fprintf(fid, '    Mean Total Time: %.3fs ± %.3fs\n', ...
                mean(planner_summary.total_time_seconds), std(planner_summary.total_time_seconds));
        fprintf(fid, '    Mean Iterations: %.1f ± %.1f\n', ...
                mean(planner_summary.final_iterations), std(planner_summary.final_iterations));
        fprintf(fid, '    Mean Final Tree Size: %.0f ± %.0f\n', ...
                mean(planner_summary.final_tree_size), std(planner_summary.final_tree_size));
        fprintf(fid, '    Best Cost: %.3f\n', min(planner_summary.final_cost));
        fprintf(fid, '    Worst Cost: %.3f\n\n', max(planner_summary.final_cost));
    end

    fprintf(fid, '\n');
end

fclose(fid);
fprintf('✓ Saved: %s\n', report_file);

%% Summary Table
fprintf('\n');
fprintf('Summary Statistics by Environment and Planner\n');
fprintf('==============================================\n\n');

for env_idx = 1:length(environments)
    env = environments{env_idx};
    fprintf('Environment: %s\n', env);

    env_mask = strcmp(summary_data.environment, env);
    env_summary = summary_data(env_mask, :);

    % Create summary table
    T = table();
    for p_idx = 1:length(planner_types)
        planner = planner_types{p_idx};
        planner_mask = strcmp(env_summary.planner_type, planner);
        planner_summary = env_summary(planner_mask, :);

        if isempty(planner_summary)
            continue;
        end

        row = table();
        row.Planner = {planner};
        row.Runs = height(planner_summary);
        row.MeanCost = mean(planner_summary.final_cost);
        row.StdCost = std(planner_summary.final_cost);
        row.MeanTime = mean(planner_summary.total_time_seconds);
        row.StdTime = std(planner_summary.total_time_seconds);
        row.MeanIters = mean(planner_summary.final_iterations);
        row.MeanTreeSize = mean(planner_summary.final_tree_size);

        T = [T; row];
    end

    disp(T);
    fprintf('\n');
end

%% Final Summary
fprintf('═══════════════════════════════════════════\n');
fprintf('All plots saved to: %s\n', OUTPUT_DIR);
fprintf('═══════════════════════════════════════════\n');
fprintf('\nGenerated files:\n');
fprintf('  - cost_convergence_{environment}.png/.fig\n');
fprintf('  - summary_statistics.png/.fig\n');
fprintf('  - benchmark_report.txt\n');
fprintf('\nDone!\n');
