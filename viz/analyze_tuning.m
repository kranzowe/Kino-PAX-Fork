close all
clc
clear all

try
    % Define root directory
    rootDir = '/home/owkr8158/Kino-PAX-Fork';

    fprintf('=== ReKinoLite Tuning Analysis ===\n');
    fprintf('Root directory: %s\n', rootDir);

    % Read tuning results
    resultsPath = fullfile(rootDir, 'build/Data/Tuning/tuning_results.csv');
    fprintf('Reading results from: %s\n', resultsPath);

    if ~exist(resultsPath, 'file')
        error('Results file not found! Please run rekino_lite_tuning first.');
    end

    data = readtable(resultsPath);
    fprintf('Loaded %d configurations\n', height(data));

    % Create output directory
    figsDir = fullfile(rootDir, 'viz/figs/tuning');
    if ~exist(figsDir, 'dir')
        fprintf('Creating figs directory: %s\n', figsDir);
        mkdir(figsDir);
    end

    % Get unique values
    samplesValues = unique(data.samplesPerThread);
    epsilonValues = unique(data.epsilonGreedy);

    fprintf('Samples per thread values: %s\n', mat2str(samplesValues'));
    fprintf('Epsilon greedy values: %s\n', mat2str(epsilonValues'));
    fprintf('\n');

    %% 1. Success Rate Heatmap
    fprintf('Generating success rate heatmap...\n');
    fig = figure('Position', [100, 100, 1000, 800], 'Visible', 'off');

    % Create pivot table for success rate
    successMatrix = zeros(length(epsilonValues), length(samplesValues));
    for i = 1:length(epsilonValues)
        for j = 1:length(samplesValues)
            idx = data.epsilonGreedy == epsilonValues(i) & data.samplesPerThread == samplesValues(j);
            if any(idx)
                successMatrix(i, j) = data.successRate(idx) * 100;
            end
        end
    end

    imagesc(samplesValues, epsilonValues, successMatrix);
    colormap(jet);
    colorbar('Label', 'Success Rate (%)');
    set(gca, 'YDir', 'normal');
    xlabel('Samples Per Thread', 'FontSize', 12);
    ylabel('Epsilon Greedy', 'FontSize', 12);
    title('Success Rate by Parameters', 'FontSize', 14, 'FontWeight', 'bold');

    % Add text annotations
    for i = 1:length(epsilonValues)
        for j = 1:length(samplesValues)
            text(samplesValues(j), epsilonValues(i), sprintf('%.1f', successMatrix(i, j)), ...
                'HorizontalAlignment', 'center', 'Color', 'white', 'FontWeight', 'bold');
        end
    end

    filename1 = fullfile(figsDir, 'success_rate_heatmap.jpg');
    print(gcf, filename1, '-djpeg', '-r300');
    fprintf('Saved: %s\n', filename1);
    close(gcf);

    %% 2. Average Time Heatmap
    fprintf('Generating execution time heatmap...\n');
    fig = figure('Position', [100, 100, 1000, 800], 'Visible', 'off');

    % Create pivot table for average time
    timeMatrix = zeros(length(epsilonValues), length(samplesValues));
    for i = 1:length(epsilonValues)
        for j = 1:length(samplesValues)
            idx = data.epsilonGreedy == epsilonValues(i) & data.samplesPerThread == samplesValues(j);
            if any(idx)
                timeMatrix(i, j) = data.avgTime(idx);
            end
        end
    end

    imagesc(samplesValues, epsilonValues, timeMatrix);
    colormap(flipud(jet));
    colorbar('Label', 'Avg Time (s)');
    set(gca, 'YDir', 'normal');
    xlabel('Samples Per Thread', 'FontSize', 12);
    ylabel('Epsilon Greedy', 'FontSize', 12);
    title('Average Execution Time by Parameters', 'FontSize', 14, 'FontWeight', 'bold');

    % Add text annotations
    for i = 1:length(epsilonValues)
        for j = 1:length(samplesValues)
            text(samplesValues(j), epsilonValues(i), sprintf('%.3f', timeMatrix(i, j)), ...
                'HorizontalAlignment', 'center', 'Color', 'white', 'FontWeight', 'bold', 'FontSize', 8);
        end
    end

    filename2 = fullfile(figsDir, 'avg_time_heatmap.jpg');
    print(gcf, filename2, '-djpeg', '-r300');
    fprintf('Saved: %s\n', filename2);
    close(gcf);

    %% 3. Effect of Samples Per Thread
    fprintf('Generating samples effect plot...\n');
    fig = figure('Position', [100, 100, 1400, 600], 'Visible', 'off');

    subplot(1, 2, 1);
    hold on;
    colors = lines(length(epsilonValues));
    for i = 1:length(epsilonValues)
        eps = epsilonValues(i);
        subset = data(data.epsilonGreedy == eps, :);
        plot(subset.samplesPerThread, subset.successRate * 100, '-o', ...
            'LineWidth', 2, 'MarkerSize', 8, 'Color', colors(i, :), ...
            'DisplayName', sprintf('ε=%.1f', eps));
    end
    xlabel('Samples Per Thread', 'FontSize', 12);
    ylabel('Success Rate (%)', 'FontSize', 12);
    title('Success Rate vs Samples Per Thread', 'FontSize', 13, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on;
    hold off;

    subplot(1, 2, 2);
    hold on;
    for i = 1:length(epsilonValues)
        eps = epsilonValues(i);
        subset = data(data.epsilonGreedy == eps, :);
        plot(subset.samplesPerThread, subset.avgTime, '-o', ...
            'LineWidth', 2, 'MarkerSize', 8, 'Color', colors(i, :), ...
            'DisplayName', sprintf('ε=%.1f', eps));
    end
    xlabel('Samples Per Thread', 'FontSize', 12);
    ylabel('Average Time (s)', 'FontSize', 12);
    title('Execution Time vs Samples Per Thread', 'FontSize', 13, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on;
    hold off;

    filename3 = fullfile(figsDir, 'samples_effect.jpg');
    print(gcf, filename3, '-djpeg', '-r300');
    fprintf('Saved: %s\n', filename3);
    close(gcf);

    %% 4. Effect of Epsilon Greedy
    fprintf('Generating epsilon effect plot...\n');
    fig = figure('Position', [100, 100, 1400, 600], 'Visible', 'off');

    subplot(1, 2, 1);
    hold on;
    colors = lines(length(samplesValues));
    for i = 1:length(samplesValues)
        samp = samplesValues(i);
        subset = data(data.samplesPerThread == samp, :);
        plot(subset.epsilonGreedy, subset.successRate * 100, '-o', ...
            'LineWidth', 2, 'MarkerSize', 8, 'Color', colors(i, :), ...
            'DisplayName', sprintf('%d samples', samp));
    end
    xlabel('Epsilon Greedy', 'FontSize', 12);
    ylabel('Success Rate (%)', 'FontSize', 12);
    title('Success Rate vs Epsilon Greedy', 'FontSize', 13, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on;
    hold off;

    subplot(1, 2, 2);
    hold on;
    for i = 1:length(samplesValues)
        samp = samplesValues(i);
        subset = data(data.samplesPerThread == samp, :);
        plot(subset.epsilonGreedy, subset.avgTime, '-o', ...
            'LineWidth', 2, 'MarkerSize', 8, 'Color', colors(i, :), ...
            'DisplayName', sprintf('%d samples', samp));
    end
    xlabel('Epsilon Greedy', 'FontSize', 12);
    ylabel('Average Time (s)', 'FontSize', 12);
    title('Execution Time vs Epsilon Greedy', 'FontSize', 13, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on;
    hold off;

    filename4 = fullfile(figsDir, 'epsilon_effect.jpg');
    print(gcf, filename4, '-djpeg', '-r300');
    fprintf('Saved: %s\n', filename4);
    close(gcf);

    %% 5. Pareto Front
    fprintf('Generating Pareto front plot...\n');
    fig = figure('Position', [100, 100, 1000, 800], 'Visible', 'off');
    hold on;

    markers = {'o', 's', '^', 'd'};
    colors = lines(length(samplesValues));

    for i = 1:length(samplesValues)
        samp = samplesValues(i);
        for j = 1:length(epsilonValues)
            eps = epsilonValues(j);
            idx = data.samplesPerThread == samp & data.epsilonGreedy == eps;
            if any(idx)
                markerIdx = mod(j-1, length(markers)) + 1;
                scatter(data.avgTime(idx), data.successRate(idx) * 100, ...
                    150, colors(i, :), markers{markerIdx}, 'filled', ...
                    'DisplayName', sprintf('s=%d, ε=%.1f', samp, eps));
            end
        end
    end

    xlabel('Average Time (s)', 'FontSize', 12);
    ylabel('Success Rate (%)', 'FontSize', 12);
    title('Pareto Front: Success Rate vs Execution Time', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'eastoutside', 'FontSize', 8);
    grid on;
    hold off;

    filename5 = fullfile(figsDir, 'pareto_front.jpg');
    print(gcf, filename5, '-djpeg', '-r300');
    fprintf('Saved: %s\n', filename5);
    close(gcf);

    %% Print Summary Statistics
    fprintf('\n=== Top 5 Configurations by Success Rate ===\n');
    [~, sortIdx] = sort(data.successRate, 'descend');
    for i = 1:min(5, height(data))
        idx = sortIdx(i);
        fprintf('  samples=%2d, ε=%.2f: %.1f%% success, %.3fs avg time\n', ...
            data.samplesPerThread(idx), data.epsilonGreedy(idx), ...
            data.successRate(idx)*100, data.avgTime(idx));
    end

    fprintf('\n=== Top 5 Configurations by Speed (among >95%% success) ===\n');
    highSuccess = data(data.successRate >= 0.95, :);
    if height(highSuccess) > 0
        [~, sortIdx] = sort(highSuccess.avgTime, 'ascend');
        for i = 1:min(5, height(highSuccess))
            idx = sortIdx(i);
            fprintf('  samples=%2d, ε=%.2f: %.1f%% success, %.3fs avg time\n', ...
                highSuccess.samplesPerThread(idx), highSuccess.epsilonGreedy(idx), ...
                highSuccess.successRate(idx)*100, highSuccess.avgTime(idx));
        end
    else
        fprintf('  No configurations achieved >= 95%% success rate\n');
    end

    fprintf('\n=== ANALYSIS COMPLETE ===\n');
    fprintf('Total configurations tested: %d\n', height(data));
    fprintf('All figures saved to: %s\n', figsDir);

catch ME
    fprintf('ERROR OCCURRED!\n');
    fprintf('Error: %s\n', ME.message);
    fprintf('Error identifier: %s\n', ME.identifier);
    fprintf('Stack trace:\n');
    for k = 1:length(ME.stack)
        fprintf('  File: %s\n', ME.stack(k).file);
        fprintf('  Function: %s, Line: %d\n', ME.stack(k).name, ME.stack(k).line);
    end
    exit(1);
end
