%% Planner Comparison Benchmark Visualization
% Reads per-iteration CSVs and summary CSV from PlannerComparisonBenchmark
% and generates plots comparing KPAX, PruneKPAX, and KinoPaxPlus.
%
% Output files are located in: Data/Benchmarks/PlannerComparison/
%   Per-iteration: {env}_{planner}_run{n}.csv
%   Summary:       planner_comparison_{timestamp}_summary.csv

clear; clc; close all;

%% --- Configuration ---
dataDir = '../Data/Benchmarks/PlannerComparison';

planners     = {'KPAX', 'PruneKPAX', 'KinoPaxPlus'};
environments = {'Empty', 'House', 'NarrowPassage', 'Trees'};
envLabels    = {'Empty', 'House', 'Narrow Passage', 'Trees'};

plannerColors = [0.2 0.4 0.8;    % KPAX - blue
                 0.9 0.5 0.1;    % PruneKPAX - orange
                 0.2 0.7 0.3];   % KinoPaxPlus - green

plannerMarkers = {'o', 's', 'd'};

numRuns = 10;

%% --- Load Summary CSV ---
summaryFiles = dir(fullfile(dataDir, 'planner_comparison_*_summary.csv'));
if isempty(summaryFiles)
    error('No summary CSV found in %s. Run the benchmark first.', dataDir);
end
% Use the most recent summary file
[~, idx] = max([summaryFiles.datenum]);
summaryPath = fullfile(dataDir, summaryFiles(idx).name);
fprintf('Loading summary: %s\n', summaryPath);
summaryTable = readtable(summaryPath);

%% --- Load Per-Iteration Data ---
% Structure: iterData{envIdx}{plannerIdx}{runIdx} = table
iterData = cell(length(environments), length(planners));

for ei = 1:length(environments)
    for pi = 1:length(planners)
        runs = cell(1, numRuns);
        for ri = 0:(numRuns - 1)
            fname = sprintf('%s_%s_run%d.csv', environments{ei}, planners{pi}, ri);
            fpath = fullfile(dataDir, fname);
            if isfile(fpath)
                runs{ri + 1} = readtable(fpath);
            end
        end
        iterData{ei, pi} = runs;
    end
end

fprintf('Data loaded for %d environments x %d planners.\n', length(environments), length(planners));

%% ======================================================================
%  FIGURE 1: Frontier Size (Nodes Propagated) vs Iteration
%  Per-environment subplots, mean +/- std across runs
%  ======================================================================
figure('Name', 'Frontier Size vs Iteration', 'Position', [50 50 1400 900]);

for ei = 1:length(environments)
    subplot(2, 2, ei); hold on;

    for pi = 1:length(planners)
        runs = iterData{ei, pi};

        % Collect all frontier sizes aligned by iteration
        maxIter = 0;
        for ri = 1:numRuns
            if ~isempty(runs{ri})
                maxIter = max(maxIter, max(runs{ri}.iteration));
            end
        end
        if maxIter == 0, continue; end

        allFrontier = NaN(numRuns, maxIter);
        for ri = 1:numRuns
            if ~isempty(runs{ri})
                iters = runs{ri}.iteration;
                allFrontier(ri, iters) = runs{ri}.frontier_size;
            end
        end

        meanF = nanmean(allFrontier, 1);
        stdF  = nanstd(allFrontier, 0, 1);
        itrVec = 1:maxIter;

        % Shaded region for std
        validIdx = ~isnan(meanF);
        xFill = [itrVec(validIdx), fliplr(itrVec(validIdx))];
        yFill = [meanF(validIdx) + stdF(validIdx), fliplr(meanF(validIdx) - stdF(validIdx))];
        yFill = max(yFill, 0);
        fill(xFill, yFill, plannerColors(pi, :), ...
            'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        plot(itrVec(validIdx), meanF(validIdx), '-', ...
            'Color', plannerColors(pi, :), 'LineWidth', 1.8, ...
            'DisplayName', planners{pi});
    end

    xlabel('Iteration');
    ylabel('Frontier Size');
    title(envLabels{ei});
    legend('Location', 'best');
    grid on;
    set(gca, 'FontSize', 10);
end
sgtitle('Frontier Size (Nodes Propagated) per Iteration', 'FontSize', 14, 'FontWeight', 'bold');

%% ======================================================================
%  FIGURE 2: Best Cost vs Iteration
%  Per-environment subplots, mean across runs
%  ======================================================================
figure('Name', 'Best Cost vs Iteration', 'Position', [100 100 1400 900]);

for ei = 1:length(environments)
    subplot(2, 2, ei); hold on;

    for pi = 1:length(planners)
        runs = iterData{ei, pi};

        maxIter = 0;
        for ri = 1:numRuns
            if ~isempty(runs{ri})
                maxIter = max(maxIter, max(runs{ri}.iteration));
            end
        end
        if maxIter == 0, continue; end

        allCost = NaN(numRuns, maxIter);
        for ri = 1:numRuns
            if ~isempty(runs{ri})
                iters = runs{ri}.iteration;
                allCost(ri, iters) = runs{ri}.best_cost;
            end
        end

        % Replace Inf with NaN for plotting
        allCost(isinf(allCost)) = NaN;

        meanC = nanmean(allCost, 1);
        stdC  = nanstd(allCost, 0, 1);
        itrVec = 1:maxIter;

        validIdx = ~isnan(meanC);
        if any(validIdx)
            xFill = [itrVec(validIdx), fliplr(itrVec(validIdx))];
            yFill = [meanC(validIdx) + stdC(validIdx), fliplr(meanC(validIdx) - stdC(validIdx))];
            fill(xFill, yFill, plannerColors(pi, :), ...
                'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            plot(itrVec(validIdx), meanC(validIdx), '-', ...
                'Color', plannerColors(pi, :), 'LineWidth', 1.8, ...
                'DisplayName', planners{pi});
        end
    end

    xlabel('Iteration');
    ylabel('Best Solution Cost');
    title(envLabels{ei});
    legend('Location', 'best');
    grid on;
    set(gca, 'FontSize', 10);
end
sgtitle('Best Solution Cost over Iterations', 'FontSize', 14, 'FontWeight', 'bold');

%% ======================================================================
%  FIGURE 3: Best Cost vs Elapsed Time (ms)
%  Per-environment subplots
%  ======================================================================
figure('Name', 'Best Cost vs Time', 'Position', [150 150 1400 900]);

for ei = 1:length(environments)
    subplot(2, 2, ei); hold on;

    for pi = 1:length(planners)
        runs = iterData{ei, pi};

        % Plot each run as a thin line, and overlay mean
        for ri = 1:numRuns
            if ~isempty(runs{ri})
                costs = runs{ri}.best_cost;
                costs(isinf(costs)) = NaN;
                t = runs{ri}.elapsed_time_ms;
                plot(t, costs, '-', 'Color', [plannerColors(pi,:), 0.2], ...
                    'LineWidth', 0.5, 'HandleVisibility', 'off');
            end
        end

        % Compute mean over time bins (approximate since times differ per run)
        % Use first run's time vector as reference
        if ~isempty(runs{1})
            refTime = runs{1}.elapsed_time_ms;
            allCostTime = NaN(numRuns, length(refTime));
            allCostTime(1, :) = runs{1}.best_cost;
            for ri = 2:numRuns
                if ~isempty(runs{ri})
                    allCostTime(ri, :) = interp1(runs{ri}.elapsed_time_ms, ...
                        runs{ri}.best_cost, refTime, 'previous', NaN);
                end
            end
            allCostTime(isinf(allCostTime)) = NaN;
            meanCT = nanmean(allCostTime, 1);
            validIdx = ~isnan(meanCT);
            if any(validIdx)
                plot(refTime(validIdx), meanCT(validIdx), '-', ...
                    'Color', plannerColors(pi,:), 'LineWidth', 2.0, ...
                    'DisplayName', planners{pi});
            end
        end
    end

    xlabel('Elapsed Time (ms)');
    ylabel('Best Solution Cost');
    title(envLabels{ei});
    legend('Location', 'best');
    grid on;
    set(gca, 'FontSize', 10);
end
sgtitle('Solution Cost Convergence over Time', 'FontSize', 14, 'FontWeight', 'bold');

%% ======================================================================
%  FIGURE 4: Tree Size Growth vs Iteration
%  ======================================================================
figure('Name', 'Tree Size vs Iteration', 'Position', [200 50 1400 900]);

for ei = 1:length(environments)
    subplot(2, 2, ei); hold on;

    for pi = 1:length(planners)
        runs = iterData{ei, pi};

        maxIter = 0;
        for ri = 1:numRuns
            if ~isempty(runs{ri})
                maxIter = max(maxIter, max(runs{ri}.iteration));
            end
        end
        if maxIter == 0, continue; end

        allTree = NaN(numRuns, maxIter);
        for ri = 1:numRuns
            if ~isempty(runs{ri})
                iters = runs{ri}.iteration;
                allTree(ri, iters) = runs{ri}.tree_size;
            end
        end

        meanT = nanmean(allTree, 1);
        itrVec = 1:maxIter;
        validIdx = ~isnan(meanT);

        plot(itrVec(validIdx), meanT(validIdx), '-', ...
            'Color', plannerColors(pi, :), 'LineWidth', 1.8, ...
            'DisplayName', planners{pi});
    end

    xlabel('Iteration');
    ylabel('Tree Size');
    title(envLabels{ei});
    legend('Location', 'best');
    grid on;
    set(gca, 'FontSize', 10);
end
sgtitle('Tree Size Growth over Iterations', 'FontSize', 14, 'FontWeight', 'bold');

%% ======================================================================
%  FIGURE 5: Summary Bar Charts
%  Left: First Solution Iteration, Right: Final Best Cost
%  ======================================================================
figure('Name', 'Summary Statistics', 'Position', [250 100 1400 500]);

% --- First Solution Iteration ---
subplot(1, 3, 1); hold on;

barData = NaN(length(environments), length(planners));
barErr  = NaN(length(environments), length(planners));

for ei = 1:length(environments)
    for pi = 1:length(planners)
        mask = strcmp(summaryTable.environment, environments{ei}) & ...
               strcmp(summaryTable.planner, planners{pi});
        vals = summaryTable.first_sol_iteration(mask);
        vals(vals < 0) = NaN;  % -1 means no solution
        barData(ei, pi) = nanmean(vals);
        barErr(ei, pi)  = nanstd(vals);
    end
end

b = bar(barData, 'grouped');
for pi = 1:length(planners)
    b(pi).FaceColor = plannerColors(pi, :);
end

% Add error bars
ngroups = size(barData, 1);
nbars = size(barData, 2);
groupwidth = min(0.8, nbars / (nbars + 1.5));
for pi = 1:nbars
    x = (1:ngroups) - groupwidth/2 + (2*pi - 1) * groupwidth / (2*nbars);
    errorbar(x, barData(:, pi), barErr(:, pi), 'k.', 'LineWidth', 1, 'HandleVisibility', 'off');
end

set(gca, 'XTickLabel', envLabels, 'FontSize', 10);
ylabel('Iteration');
title('First Solution Iteration');
legend(planners, 'Location', 'best');
grid on;

% --- Final Best Cost ---
subplot(1, 3, 2); hold on;

barData2 = NaN(length(environments), length(planners));
barErr2  = NaN(length(environments), length(planners));

for ei = 1:length(environments)
    for pi = 1:length(planners)
        mask = strcmp(summaryTable.environment, environments{ei}) & ...
               strcmp(summaryTable.planner, planners{pi});
        vals = summaryTable.final_best_cost(mask);
        vals(isinf(vals)) = NaN;
        barData2(ei, pi) = nanmean(vals);
        barErr2(ei, pi)  = nanstd(vals);
    end
end

b2 = bar(barData2, 'grouped');
for pi = 1:length(planners)
    b2(pi).FaceColor = plannerColors(pi, :);
end

ngroups = size(barData2, 1);
nbars = size(barData2, 2);
groupwidth = min(0.8, nbars / (nbars + 1.5));
for pi = 1:nbars
    x = (1:ngroups) - groupwidth/2 + (2*pi - 1) * groupwidth / (2*nbars);
    errorbar(x, barData2(:, pi), barErr2(:, pi), 'k.', 'LineWidth', 1, 'HandleVisibility', 'off');
end

set(gca, 'XTickLabel', envLabels, 'FontSize', 10);
ylabel('Cost');
title('Final Best Solution Cost');
legend(planners, 'Location', 'best');
grid on;

% --- Total Execution Time ---
subplot(1, 3, 3); hold on;

barData3 = NaN(length(environments), length(planners));
barErr3  = NaN(length(environments), length(planners));

for ei = 1:length(environments)
    for pi = 1:length(planners)
        mask = strcmp(summaryTable.environment, environments{ei}) & ...
               strcmp(summaryTable.planner, planners{pi});
        vals = summaryTable.total_time_s(mask);
        barData3(ei, pi) = nanmean(vals);
        barErr3(ei, pi)  = nanstd(vals);
    end
end

b3 = bar(barData3, 'grouped');
for pi = 1:length(planners)
    b3(pi).FaceColor = plannerColors(pi, :);
end

ngroups = size(barData3, 1);
nbars = size(barData3, 2);
groupwidth = min(0.8, nbars / (nbars + 1.5));
for pi = 1:nbars
    x = (1:ngroups) - groupwidth/2 + (2*pi - 1) * groupwidth / (2*nbars);
    errorbar(x, barData3(:, pi), barErr3(:, pi), 'k.', 'LineWidth', 1, 'HandleVisibility', 'off');
end

set(gca, 'XTickLabel', envLabels, 'FontSize', 10);
ylabel('Time (s)');
title('Total Execution Time');
legend(planners, 'Location', 'best');
grid on;

sgtitle('Planner Comparison Summary (mean +/- std over runs)', 'FontSize', 14, 'FontWeight', 'bold');

%% ======================================================================
%  FIGURE 6: Solution Success Rate
%  ======================================================================
figure('Name', 'Solution Success Rate', 'Position', [300 150 800 400]);

successRate = NaN(length(environments), length(planners));

for ei = 1:length(environments)
    for pi = 1:length(planners)
        mask = strcmp(summaryTable.environment, environments{ei}) & ...
               strcmp(summaryTable.planner, planners{pi});
        vals = summaryTable.first_sol_iteration(mask);
        successRate(ei, pi) = sum(vals > 0) / length(vals) * 100;
    end
end

b4 = bar(successRate, 'grouped');
for pi = 1:length(planners)
    b4(pi).FaceColor = plannerColors(pi, :);
end

set(gca, 'XTickLabel', envLabels, 'FontSize', 11);
ylabel('Success Rate (%)');
ylim([0 110]);
title('Solution Success Rate Across Environments', 'FontSize', 13);
legend(planners, 'Location', 'best');
grid on;

% Add percentage labels on bars
for pi = 1:length(planners)
    xtips = b4(pi).XEndPoints;
    ytips = b4(pi).YEndPoints;
    labels = string(round(successRate(:, pi))) + "%";
    text(xtips, ytips + 2, labels, 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', 'FontSize', 9, 'FontWeight', 'bold');
end

fprintf('\nAll figures generated.\n');
