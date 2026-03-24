%% KinoPaxPlus Delta Benchmark Visualization
% Reads per-iteration CSVs produced by kinopaxplus_delta_benchmark.cu
% (run via run_delta_benchmark.sh) and generates convergence plots
% comparing different region discretizations (delta).
%
% Output directory: Data/Benchmarks/KinoPaxPlusDelta/
%   Per-iteration: Trees_delta{label}_run{n}.csv
%   Summary:       delta_benchmark_{timestamp}_summary.csv
%
% Per-iteration columns: iteration, frontier_size, tree_size, elapsed_time_ms, best_cost

clear; clc; close all;

%% --- Configuration ---
dataDir = '../build/Data/Benchmarks/KinoPaxPlusDelta';

deltas      = {'large', 'med_large', 'med_small', 'small'};
deltaLabels = {'Large-\delta (27k)', ...
               'Med-Large (216k)', ...
               'Med-Small (1.7M)', ...
               'Small-\delta (10M)'};

deltaColors = [0.2 0.4 0.8;    % large     - blue
               0.9 0.5 0.1;    % med_large - orange
               0.2 0.7 0.3;    % med_small - green
               0.8 0.2 0.2];   % small     - red

deltaStyles = {'-', '--', '-.', ':'};

numRuns = 10;
environment = 'Trees';

% Model 1 MAX_FLOAT is 1e6 (not 1e38 like Model 3)
MAX_FLOAT_THRESH = 1e5;

%% --- Locate the most recent summary file ---
summaryFiles = dir(fullfile(dataDir, 'delta_benchmark_*_summary.csv'));
if isempty(summaryFiles)
    warning('No summary CSV found in %s. Summary bar charts will be skipped.', dataDir);
    summaryTable = [];
else
    [~, idx] = max([summaryFiles.datenum]);
    summaryPath = fullfile(dataDir, summaryFiles(idx).name);
    fprintf('Loading summary: %s\n', summaryPath);
    summaryTable = readtable(summaryPath);
end

%% --- Load Per-Iteration Data ---
% Structure: iterData{deltaIdx}{runIdx} = table
iterData = cell(1, length(deltas));

for di = 1:length(deltas)
    runs = cell(1, numRuns);
    for ri = 0:(numRuns - 1)
        fname = sprintf('%s_delta%s_run%d.csv', environment, deltas{di}, ri);
        fpath = fullfile(dataDir, fname);
        if isfile(fpath)
            runs{ri + 1} = readtable(fpath);
        end
    end
    iterData{di} = runs;
end

fprintf('Data loaded for %d delta configurations.\n', length(deltas));

%% --- Helper: find first solution iteration ---
function firstIter = firstSolutionIter(tbl, thresh)
    solIdx = find(tbl.best_cost < thresh, 1, 'first');
    if isempty(solIdx)
        firstIter = -1;
    else
        firstIter = tbl.iteration(solIdx);
    end
end

%% ======================================================================
%  FIGURE 1: Best Cost vs Iteration
%  Mean +/- std across runs for each delta
%  ======================================================================
figure('Name', 'Best Cost vs Iteration', 'Position', [50 50 900 600]);
hold on;

for di = 1:length(deltas)
    runs = iterData{di};

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

    % Replace MAX_FLOAT values with NaN
    allCost(allCost > MAX_FLOAT_THRESH) = NaN;

    meanC  = mean(allCost, 1, 'omitnan');
    stdC   = std(allCost, 0, 1, 'omitnan');
    itrVec = 1:maxIter;

    validIdx = ~isnan(meanC);
    if any(validIdx)
        xFill = [itrVec(validIdx), fliplr(itrVec(validIdx))];
        yFill = [meanC(validIdx) + stdC(validIdx), fliplr(meanC(validIdx) - stdC(validIdx))];
        fill(xFill, yFill, deltaColors(di, :), ...
            'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        plot(itrVec(validIdx), meanC(validIdx), deltaStyles{di}, ...
            'Color', deltaColors(di, :), 'LineWidth', 1.8, ...
            'DisplayName', deltaLabels{di});
    end
end

xlabel('Iteration');
ylabel('Path Cost (workspace distance)');
title(sprintf('Best Cost vs Iteration — %s Environment', environment));
legend('Location', 'best', 'FontSize', 7);
grid on;
set(gca, 'FontSize', 10);

%% ======================================================================
%  FIGURE 2: Best Cost vs Elapsed Time
%  Individual runs as thin lines, mean as bold line
%  ======================================================================
figure('Name', 'Best Cost vs Time', 'Position', [100 100 900 600]);
hold on;

for di = 1:length(deltas)
    runs = iterData{di};

    % Plot each run as a thin line
    for ri = 1:numRuns
        if ~isempty(runs{ri})
            costs = runs{ri}.best_cost;
            costs(costs > MAX_FLOAT_THRESH) = NaN;
            t = runs{ri}.elapsed_time_ms;  % already cumulative
            plot(t, costs, '-', 'Color', [deltaColors(di,:), 0.2], ...
                'LineWidth', 0.5, 'HandleVisibility', 'off');
        end
    end

    % Overlay mean using first run's time axis as reference
    if ~isempty(runs{1})
        refTime = runs{1}.elapsed_time_ms;
        refCost = runs{1}.best_cost;
        refCost(refCost > MAX_FLOAT_THRESH) = NaN;
        allCostTime = NaN(numRuns, length(refTime));
        allCostTime(1, :) = refCost;
        for ri = 2:numRuns
            if ~isempty(runs{ri})
                riCost = runs{ri}.best_cost;
                riCost(riCost > MAX_FLOAT_THRESH) = NaN;
                riTime = runs{ri}.elapsed_time_ms;
                allCostTime(ri, :) = interp1(riTime, riCost, ...
                    refTime, 'previous', NaN);
            end
        end
        meanCT = mean(allCostTime, 1, 'omitnan');
        validIdx = ~isnan(meanCT);
        if any(validIdx)
            plot(refTime(validIdx), meanCT(validIdx), deltaStyles{di}, ...
                'Color', deltaColors(di,:), 'LineWidth', 2.0, ...
                'DisplayName', deltaLabels{di});
        end
    end
end

xlabel('Elapsed Time (ms)');
ylabel('Path Cost (workspace distance)');
title(sprintf('Cost Convergence over Wall Time — %s Environment', environment));
legend('Location', 'best', 'FontSize', 7);
grid on;
set(gca, 'FontSize', 10);

%% ======================================================================
%  FIGURE 3: Summary Bar Charts
%  Left: First Solution Iteration, Centre: Final Best Cost, Right: Total Time
%  ======================================================================
if ~isempty(summaryTable)
    figure('Name', 'Summary Statistics', 'Position', [150 150 1400 450]);

    nDelta = length(deltas);

    % --- First Solution Iteration (from per-iteration data) ---
    subplot(1, 3, 1); hold on;

    barData = NaN(1, nDelta);
    barErr  = NaN(1, nDelta);

    for di = 1:nDelta
        runs = iterData{di};
        vals = NaN(1, numRuns);
        for ri = 1:numRuns
            if ~isempty(runs{ri})
                vals(ri) = firstSolutionIter(runs{ri}, MAX_FLOAT_THRESH);
            end
        end
        vals(vals < 0) = NaN;
        barData(di) = mean(vals, 'omitnan');
        barErr(di)  = std(vals, 'omitnan');
    end

    b1 = bar(barData);
    b1.FaceColor = 'flat';
    for di = 1:nDelta
        b1.CData(di,:) = deltaColors(di,:);
    end
    errorbar(1:nDelta, barData, barErr, 'k.', 'LineWidth', 1);

    set(gca, 'XTick', 1:nDelta, 'XTickLabel', deltaLabels, 'FontSize', 8);
    xtickangle(25);
    ylabel('Iteration');
    title('First Solution Iteration');
    grid on;

    % --- Final Best Cost (from summary CSV) ---
    subplot(1, 3, 2); hold on;

    barData2 = NaN(1, nDelta);
    barErr2  = NaN(1, nDelta);

    for di = 1:nDelta
        mask = strcmp(summaryTable.delta_label, deltas{di});
        vals = summaryTable.final_best_cost(mask);
        vals(vals > MAX_FLOAT_THRESH) = NaN;
        barData2(di) = mean(vals, 'omitnan');
        barErr2(di)  = std(vals, 'omitnan');
    end

    b2 = bar(barData2);
    b2.FaceColor = 'flat';
    for di = 1:nDelta
        b2.CData(di,:) = deltaColors(di,:);
    end
    errorbar(1:nDelta, barData2, barErr2, 'k.', 'LineWidth', 1);

    set(gca, 'XTick', 1:nDelta, 'XTickLabel', deltaLabels, 'FontSize', 8);
    xtickangle(25);
    ylabel('Path Cost (workspace distance)');
    title('Final Best Cost');
    grid on;

    % --- Total Execution Time (from summary CSV) ---
    subplot(1, 3, 3); hold on;

    barData3 = NaN(1, nDelta);
    barErr3  = NaN(1, nDelta);

    for di = 1:nDelta
        mask = strcmp(summaryTable.delta_label, deltas{di});
        vals = summaryTable.total_time_s(mask);
        barData3(di) = mean(vals, 'omitnan');
        barErr3(di)  = std(vals, 'omitnan');
    end

    b3 = bar(barData3);
    b3.FaceColor = 'flat';
    for di = 1:nDelta
        b3.CData(di,:) = deltaColors(di,:);
    end
    errorbar(1:nDelta, barData3, barErr3, 'k.', 'LineWidth', 1);

    set(gca, 'XTick', 1:nDelta, 'XTickLabel', deltaLabels, 'FontSize', 8);
    xtickangle(25);
    ylabel('Time (s)');
    title('Total Execution Time');
    grid on;

    sgtitle(sprintf('KinoPaxPlus Delta Comparison — %s (mean \\pm std over %d runs)', ...
        environment, numRuns), 'FontSize', 12, 'FontWeight', 'bold');
end

%% ======================================================================
%  FIGURE 4: Tree Size Growth vs Iteration
%  ======================================================================
figure('Name', 'Tree Size vs Iteration', 'Position', [200 50 900 600]);
hold on;

for di = 1:length(deltas)
    runs = iterData{di};

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

    plot(itrVec(validIdx), meanT(validIdx), deltaStyles{di}, ...
        'Color', deltaColors(di, :), 'LineWidth', 1.8, ...
        'DisplayName', deltaLabels{di});
end

xlabel('Iteration');
ylabel('Tree Size');
title(sprintf('Tree Size Growth over Iterations — %s Environment', environment));
legend('Location', 'best', 'FontSize', 7);
grid on;
set(gca, 'FontSize', 10);

%% ======================================================================
%  FIGURE 5: Solution Success Rate
%  ======================================================================
figure('Name', 'Solution Success Rate', 'Position', [250 100 700 400]);

nDelta = length(deltas);
successRate = NaN(1, nDelta);

for di = 1:nDelta
    runs = iterData{di};
    nSuccess = 0;
    nTotal   = 0;
    for ri = 1:numRuns
        if ~isempty(runs{ri})
            nTotal = nTotal + 1;
            if firstSolutionIter(runs{ri}, MAX_FLOAT_THRESH) > 0
                nSuccess = nSuccess + 1;
            end
        end
    end
    if nTotal > 0
        successRate(di) = nSuccess / nTotal * 100;
    end
end

b4 = bar(successRate);
b4.FaceColor = 'flat';
for di = 1:nDelta
    b4.CData(di,:) = deltaColors(di,:);
end

set(gca, 'XTick', 1:nDelta, 'XTickLabel', deltaLabels, 'FontSize', 11);
xtickangle(25);
ylabel('Success Rate (%)');
ylim([0 110]);
title('Solution Success Rate by Delta', 'FontSize', 13);
grid on;

% Add percentage labels on bars
xtips = b4.XEndPoints;
ytips = b4.YEndPoints;
labels = string(round(successRate)) + "%";
text(xtips, ytips + 2, labels, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');

fprintf('\nAll figures generated.\n');
