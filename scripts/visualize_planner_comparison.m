%% Planner Comparison Benchmark Visualization
% Reads per-iteration CSVs and summary CSV produced by
% examples/gpu/planner_comparison_benchmark.cu and generates plots
% comparing all four planner variants.
%
% Output directory: Data/Benchmarks/PlannerComparison/
%   Summary:       planner_comparison_{timestamp}_summary.csv
%   Per-iteration: {Environment}_{Planner}_run{n}.csv
%
% Per-iteration columns: iteration, frontier_size, tree_size, elapsed_time_ms, best_cost
%   - best_cost        : best accumulated workspace path cost found so far
%   - elapsed_time_ms  : cumulative wall time since planner start (NOT per-iteration)
%   - frontier_size    : frontier size at this iteration

clear; clc; close all;

%% --- Configuration ---
dataDir = '../Data/Benchmarks/PlannerComparison';

planners = {'KPAX', ...
            'KPAX_SpatialHash', ...
            'PruneKPAX', ...
            'KinoPaxPlus'};

plannerLabels = {'KPAX (Naive Optimal)', ...
                 'KPAX + SpatialHash', ...
                 'PruneKPAX', ...
                 'KinoPaxPlus'};

environments = {'Empty', 'House', 'NarrowPassage', 'Trees'};
envLabels    = {'Empty', 'House', 'Narrow Passage', 'Trees'};

plannerColors = [0.2 0.4 0.8;    % KPAX              - blue
                 0.9 0.5 0.1;    % KPAX SpatialHash  - orange
                 0.7 0.1 0.6;    % PruneKPAX         - purple
                 0.2 0.7 0.3];   % KinoPaxPlus       - green

plannerStyles = {'-', '--', '-.', ':'};

numRuns = 10;

%% --- Locate the most recent summary file ---
summaryFiles = dir(fullfile(dataDir, 'planner_comparison_*_summary.csv'));
if isempty(summaryFiles)
    error('No summary CSV found in %s. Run the benchmark first.', dataDir);
end
[~, idx] = max([summaryFiles.datenum]);
summaryPath = fullfile(dataDir, summaryFiles(idx).name);
fprintf('Loading summary: %s\n', summaryPath);
summaryTable = readtable(summaryPath);

%% --- Load Per-Iteration Data ---
% Per-iteration filenames: {Environment}_{Planner}_run{n}.csv (no prefix)
% Structure: iterData{envIdx, plannerIdx}{runIdx} = table
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

%% --- Helper: find first solution iteration (first iter where best_cost < very large) ---
function firstIter = firstSolutionIter(tbl)
    % MAX_FLOAT ~ 3.4e38; treat anything above 1e30 as "no solution yet"
    solIdx = find(tbl.best_cost < 1e30, 1, 'first');
    if isempty(solIdx)
        firstIter = -1;
    else
        firstIter = tbl.iteration(solIdx);
    end
end

%% ======================================================================
%  FIGURE 1: Frontier Size vs Iteration
%  Per-environment subplots (2x2), mean +/- std across runs
%  ======================================================================
figure('Name', 'Frontier Size vs Iteration', 'Position', [50 50 1200 900]);

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

        allFrontier = NaN(numRuns, maxIter);
        for ri = 1:numRuns
            if ~isempty(runs{ri})
                iters = runs{ri}.iteration;
                allFrontier(ri, iters) = runs{ri}.frontier_size;
            end
        end

        meanF = mean(allFrontier, 1, 'omitnan');
        stdF  = std(allFrontier, 0, 1, 'omitnan');
        itrVec = 1:maxIter;

        validIdx = ~isnan(meanF);
        xFill = [itrVec(validIdx), fliplr(itrVec(validIdx))];
        yFill = [meanF(validIdx) + stdF(validIdx), fliplr(meanF(validIdx) - stdF(validIdx))];
        yFill = max(yFill, 0);
        fill(xFill, yFill, plannerColors(pi, :), ...
            'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        plot(itrVec(validIdx), meanF(validIdx), plannerStyles{pi}, ...
            'Color', plannerColors(pi, :), 'LineWidth', 1.8, ...
            'DisplayName', plannerLabels{pi});
    end

    xlabel('Iteration');
    ylabel('Frontier Size');
    title(envLabels{ei});
    legend('Location', 'best', 'FontSize', 7);
    grid on;
    set(gca, 'FontSize', 10);
end
sgtitle('Frontier Size per Iteration', 'FontSize', 14, 'FontWeight', 'bold');

%% ======================================================================
%  FIGURE 2: Best Cost vs Iteration
%  Per-environment subplots (2x2), mean across runs
%  ======================================================================
figure('Name', 'Best Cost vs Iteration', 'Position', [100 100 1200 900]);

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

        % Replace huge values (MAX_FLOAT ~ 3.4e38) with NaN for plotting
        allCost(allCost > 1e30) = NaN;

        meanC = mean(allCost, 1, 'omitnan');
        stdC  = std(allCost, 0, 1, 'omitnan');
        itrVec = 1:maxIter;

        validIdx = ~isnan(meanC);
        if any(validIdx)
            xFill = [itrVec(validIdx), fliplr(itrVec(validIdx))];
            yFill = [meanC(validIdx) + stdC(validIdx), fliplr(meanC(validIdx) - stdC(validIdx))];
            fill(xFill, yFill, plannerColors(pi, :), ...
                'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            plot(itrVec(validIdx), meanC(validIdx), plannerStyles{pi}, ...
                'Color', plannerColors(pi, :), 'LineWidth', 1.8, ...
                'DisplayName', plannerLabels{pi});
        end
    end

    xlabel('Iteration');
    ylabel('Path Cost (workspace distance)');
    title(envLabels{ei});
    legend('Location', 'best', 'FontSize', 7);
    grid on;
    set(gca, 'FontSize', 10);
end
sgtitle('Best Path Cost over Iterations (accumulated root-to-goal distance)', ...
    'FontSize', 13, 'FontWeight', 'bold');

%% ======================================================================
%  FIGURE 3: Best Cost vs Cumulative Elapsed Time (ms)
%  elapsed_time_ms is ALREADY cumulative — no cumsum needed.
%  Per-environment subplots (2x2)
%  ======================================================================
figure('Name', 'Best Cost vs Time', 'Position', [150 150 1200 900]);

for ei = 1:length(environments)
    subplot(2, 2, ei); hold on;

    for pi = 1:length(planners)
        runs = iterData{ei, pi};

        % Plot each run as a thin line
        for ri = 1:numRuns
            if ~isempty(runs{ri})
                costs = runs{ri}.best_cost;
                costs(costs > 1e30) = NaN;
                t = runs{ri}.elapsed_time_ms;  % already cumulative
                plot(t, costs, '-', 'Color', [plannerColors(pi,:), 0.2], ...
                    'LineWidth', 0.5, 'HandleVisibility', 'off');
            end
        end

        % Overlay mean using first run's time axis as reference
        if ~isempty(runs{1})
            refTime = runs{1}.elapsed_time_ms;
            refCost = runs{1}.best_cost;
            refCost(refCost > 1e30) = NaN;
            allCostTime = NaN(numRuns, length(refTime));
            allCostTime(1, :) = refCost;
            for ri = 2:numRuns
                if ~isempty(runs{ri})
                    riCost = runs{ri}.best_cost;
                    riCost(riCost > 1e30) = NaN;
                    riTime = runs{ri}.elapsed_time_ms;
                    allCostTime(ri, :) = interp1(riTime, riCost, ...
                        refTime, 'previous', NaN);
                end
            end
            meanCT = mean(allCostTime, 1, 'omitnan');
            validIdx = ~isnan(meanCT);
            if any(validIdx)
                plot(refTime(validIdx), meanCT(validIdx), plannerStyles{pi}, ...
                    'Color', plannerColors(pi,:), 'LineWidth', 2.0, ...
                    'DisplayName', plannerLabels{pi});
            end
        end
    end

    xlabel('Elapsed Time (ms)');
    ylabel('Path Cost (workspace distance)');
    title(envLabels{ei});
    legend('Location', 'best', 'FontSize', 7);
    grid on;
    set(gca, 'FontSize', 10);
end
sgtitle('Path Cost Convergence over Wall Time', 'FontSize', 13, 'FontWeight', 'bold');

%% ======================================================================
%  FIGURE 4: Tree Size Growth vs Iteration
%  ======================================================================
figure('Name', 'Tree Size vs Iteration', 'Position', [200 50 1200 900]);

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

        meanT = mean(allTree, 1, 'omitnan');
        itrVec = 1:maxIter;
        validIdx = ~isnan(meanT);

        plot(itrVec(validIdx), meanT(validIdx), plannerStyles{pi}, ...
            'Color', plannerColors(pi, :), 'LineWidth', 1.8, ...
            'DisplayName', plannerLabels{pi});
    end

    xlabel('Iteration');
    ylabel('Tree Size');
    title(envLabels{ei});
    legend('Location', 'best', 'FontSize', 7);
    grid on;
    set(gca, 'FontSize', 10);
end
sgtitle('Tree Size Growth over Iterations', 'FontSize', 14, 'FontWeight', 'bold');

%% ======================================================================
%  FIGURE 5: Summary Bar Charts
%  Left: First Solution Iteration, Centre: Final Best Cost, Right: Total Time
%  ======================================================================
figure('Name', 'Summary Statistics', 'Position', [250 100 1600 500]);

nEnv = length(environments);
nPl  = length(planners);

% --- First Solution Iteration (computed from per-iteration data) ---
subplot(1, 3, 1); hold on;

barData = NaN(nEnv, nPl);
barErr  = NaN(nEnv, nPl);

for ei = 1:nEnv
    for pi = 1:nPl
        runs = iterData{ei, pi};
        vals = NaN(1, numRuns);
        for ri = 1:numRuns
            if ~isempty(runs{ri})
                vals(ri) = firstSolutionIter(runs{ri});
            end
        end
        vals(vals < 0) = NaN;  % -1 means no solution found
        barData(ei, pi) = mean(vals, 'omitnan');
        barErr(ei, pi)  = std(vals, 'omitnan');
    end
end

b = bar(barData, 'grouped');
for pi = 1:nPl
    b(pi).FaceColor = plannerColors(pi, :);
end

ngroups = size(barData, 1);
nbars = size(barData, 2);
groupwidth = min(0.8, nbars / (nbars + 1.5));
for pi = 1:nbars
    x = (1:ngroups) - groupwidth/2 + (2*pi - 1) * groupwidth / (2*nbars);
    errorbar(x, barData(:, pi), barErr(:, pi), 'k.', 'LineWidth', 1, 'HandleVisibility', 'off');
end

set(gca, 'XTickLabel', envLabels, 'FontSize', 9);
ylabel('Iteration');
title('First Solution Iteration');
legend(plannerLabels, 'Location', 'best', 'FontSize', 6);
grid on;

% --- Final Best Cost ---
subplot(1, 3, 2); hold on;

barData2 = NaN(nEnv, nPl);
barErr2  = NaN(nEnv, nPl);

for ei = 1:nEnv
    for pi = 1:nPl
        mask = strcmp(summaryTable.environment, environments{ei}) & ...
               strcmp(summaryTable.planner, planners{pi});
        vals = summaryTable.final_best_cost(mask);
        vals(vals > 1e30) = NaN;  % MAX_FLOAT = no solution
        barData2(ei, pi) = mean(vals, 'omitnan');
        barErr2(ei, pi)  = std(vals, 'omitnan');
    end
end

b2 = bar(barData2, 'grouped');
for pi = 1:nPl
    b2(pi).FaceColor = plannerColors(pi, :);
end

ngroups = size(barData2, 1);
nbars = size(barData2, 2);
groupwidth = min(0.8, nbars / (nbars + 1.5));
for pi = 1:nbars
    x = (1:ngroups) - groupwidth/2 + (2*pi - 1) * groupwidth / (2*nbars);
    errorbar(x, barData2(:, pi), barErr2(:, pi), 'k.', 'LineWidth', 1, 'HandleVisibility', 'off');
end

set(gca, 'XTickLabel', envLabels, 'FontSize', 9);
ylabel('Path Cost (workspace distance)');
title('Final Best Path Cost');
legend(plannerLabels, 'Location', 'best', 'FontSize', 6);
grid on;

% --- Total Execution Time ---
subplot(1, 3, 3); hold on;

barData3 = NaN(nEnv, nPl);
barErr3  = NaN(nEnv, nPl);

for ei = 1:nEnv
    for pi = 1:nPl
        mask = strcmp(summaryTable.environment, environments{ei}) & ...
               strcmp(summaryTable.planner, planners{pi});
        vals = summaryTable.total_time_s(mask);
        barData3(ei, pi) = mean(vals, 'omitnan');
        barErr3(ei, pi)  = std(vals, 'omitnan');
    end
end

b3 = bar(barData3, 'grouped');
for pi = 1:nPl
    b3(pi).FaceColor = plannerColors(pi, :);
end

ngroups = size(barData3, 1);
nbars = size(barData3, 2);
groupwidth = min(0.8, nbars / (nbars + 1.5));
for pi = 1:nbars
    x = (1:ngroups) - groupwidth/2 + (2*pi - 1) * groupwidth / (2*nbars);
    errorbar(x, barData3(:, pi), barErr3(:, pi), 'k.', 'LineWidth', 1, 'HandleVisibility', 'off');
end

set(gca, 'XTickLabel', envLabels, 'FontSize', 9);
ylabel('Time (s)');
title('Total Execution Time');
legend(plannerLabels, 'Location', 'best', 'FontSize', 6);
grid on;

sgtitle({'Planner Comparison Summary (mean +/- std over runs)', ...
         'Cost = accumulated workspace path length (same metric for all planners)'}, ...
         'FontSize', 12, 'FontWeight', 'bold');

%% ======================================================================
%  FIGURE 6: Solution Success Rate
%  ======================================================================
figure('Name', 'Solution Success Rate', 'Position', [300 150 900 400]);

successRate = NaN(nEnv, nPl);

for ei = 1:nEnv
    for pi = 1:nPl
        runs = iterData{ei, pi};
        nSuccess = 0;
        nTotal   = 0;
        for ri = 1:numRuns
            if ~isempty(runs{ri})
                nTotal = nTotal + 1;
                if firstSolutionIter(runs{ri}) > 0
                    nSuccess = nSuccess + 1;
                end
            end
        end
        if nTotal > 0
            successRate(ei, pi) = nSuccess / nTotal * 100;
        end
    end
end

b4 = bar(successRate, 'grouped');
for pi = 1:nPl
    b4(pi).FaceColor = plannerColors(pi, :);
end

set(gca, 'XTickLabel', envLabels, 'FontSize', 11);
ylabel('Success Rate (%)');
ylim([0 110]);
title('Solution Success Rate Across Environments', 'FontSize', 13);
legend(plannerLabels, 'Location', 'best', 'FontSize', 7);
grid on;

% Add percentage labels on bars
for pi = 1:nPl
    xtips = b4(pi).XEndPoints;
    ytips = b4(pi).YEndPoints;
    labels = string(round(successRate(:, pi))) + "%";
    text(xtips, ytips + 2, labels, 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', 'FontSize', 8, 'FontWeight', 'bold');
end

fprintf('\nAll figures generated.\n');
