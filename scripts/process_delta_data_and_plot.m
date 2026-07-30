%% KinoPaxPlus Delta Benchmark Visualization
% Reads per-iteration CSVs produced by kinopaxplus_delta_benchmark.cu
% (run via run_delta_benchmark.sh) and generates convergence plots
% comparing different region discretizations (delta) with a KPAX baseline.
%
% Output directory: Data/Benchmarks/KinoPaxPlusDelta/
%   KinoPaxPlus:  {env}_delta{label}_run{n}.csv
%   KPAX:         {env}_KPAX_delta{label}_run{n}.csv
%   PruneKPAX:    {env}_PruneKPAX_delta{label}_run{n}.csv
%   KPAXPlus:     {env}_KPAXPlus_delta{label}_run{n}.csv
%   Summary:      delta_benchmark_{timestamp}_summary.csv
%
% Per-iteration columns: iteration, frontier_size, tree_size, elapsed_time_ms, best_cost

clear; clc; close all;

%% --- Configuration ---
dataDir = '';

environments = {'house'};
envTitles    = {'House'};

deltas      = {'larger', 'large', 'med_large', 'med_small'};
deltaLabels = {'Larger-\delta (14k)', ...
               'Large-\delta (27k)', ...
               'Med-Large (216k)', ...
               'Med-Small (422k)'};

deltaColors = [0.9 0.5 0.1;    % larger    - orange
               0.2 0.7 0.3;    % large     - green
               0.8 0.2 0.2;    % med_large - red
               0.6 0.1 0.7];   % med_small - purple

deltaStyles = {'-', '--', '-.', ':'};

% KPAX baseline gets its own cool-tone palette so it reads as clearly
% separate from the optimal planners (KinoPaxPlus / KPAXPlus), which use
% the warm deltaColors above. Kept internally distinct per delta.
kpaxDeltaColors = [0.20 0.75 0.85;    % larger    - cyan
                   0.10 0.45 0.85;    % large     - blue
                   0.05 0.10 0.45;    % med_large - navy
                   0.40 0.60 0.75];   % med_small - steel blue

kpaxColor = [0.1 0.1 0.1];     % near-black for KPAX curve
kpaxFirstColor = [0.5 0.5 0.5]; % gray for reference lines
kpaxFinalColor = [0.2 0.2 0.2]; % dark gray
kpaxTimeColor  = [0.6 0.3 0.6]; % purple-ish

numRuns        = 50;   % KinoPaxPlus runs per delta
numKPAXRuns    = 50;   % KPAX baseline runs
numKPPplusRuns = 50;   % KPAXPlus runs per delta (new benchmark; dotted overlay)

% Model 1 MAX_FLOAT is 1e38
MAX_FLOAT_THRESH = 1e30;

% Box plot settings
numTimePoints = 15;   % number of time sample points for box plots
boxWidth      = 0.85;  % relative width of each box group (increased from 0.6)

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

%% ======================================================================
%  LOOP OVER ENVIRONMENTS
%  ======================================================================
figNum = 0;

for ei = 1:length(environments)
    env      = environments{ei};
    envTitle = envTitles{ei};

    fprintf('\n=== Processing environment: %s ===\n', env);

    %% --- Load KinoPaxPlus Per-Iteration Data ---
    iterData = cell(1, length(deltas));
    for di = 1:length(deltas)
        runs = cell(1, numRuns);
        for ri = 0:(numRuns - 1)
            fname = sprintf('%s_delta%s_run%d.csv', env, deltas{di}, ri);
            fpath = fullfile(dataDir, fname);
            if isfile(fpath)
                runs{ri + 1} = readtable(fpath);
            end
        end
        iterData{di} = runs;
    end
    fprintf('  KinoPaxPlus data loaded for %d delta configurations.\n', length(deltas));

    %% --- Load KPAX Per-Iteration Data (per-delta: {env}_KPAX_delta{delta}_run{n}.csv) ---
    kpaxIterData = cell(1, length(deltas));
    for di = 1:length(deltas)
        runs = cell(1, numKPAXRuns);
        for ri = 0:(numKPAXRuns - 1)
            fname = sprintf('%s_KPAX_delta%s_run%d.csv', env, deltas{di}, ri);
            fpath = fullfile(dataDir, fname);
            if isfile(fpath)
                runs{ri + 1} = readtable(fpath);
            end
        end
        kpaxIterData{di} = runs;
    end
    fprintf('  KPAX data loaded for %d delta configurations.\n', length(deltas));

    %% --- Load KPAXPlus Per-Iteration Data (new benchmark: {env}_KPAXPlus_delta{delta}_run{n}.csv) ---
    kpaxPlusIterData = cell(1, length(deltas));
    for di = 1:length(deltas)
        runs = cell(1, numKPPplusRuns);
        for ri = 0:(numKPPplusRuns - 1)
            fname = sprintf('%s_KPAXPlus_delta%s_run%d.csv', env, deltas{di}, ri);
            fpath = fullfile(dataDir, fname);
            if isfile(fpath)
                runs{ri + 1} = readtable(fpath);
            end
        end
        kpaxPlusIterData{di} = runs;
    end
    fprintf('  KPAXPlus data loaded for %d delta configurations.\n', length(deltas));

    %% ==================================================================
    %  FIGURE: Best Cost vs Iteration (mean +/- std)
    %  ==================================================================
    figNum = figNum + 1;
    figure('Name', sprintf('%s - Cost vs Iteration', envTitle), ...
           'Position', [50 50 900 600]);
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

    % --- KPAXPlus curves on iteration plot (dotted, same delta colors) ---
    for di = 1:length(deltas)
        runs = kpaxPlusIterData{di};
        maxIter = 0;
        for ri = 1:numKPPplusRuns
            if ~isempty(runs{ri}), maxIter = max(maxIter, max(runs{ri}.iteration)); end
        end
        if maxIter == 0, continue; end
        allCost = NaN(numKPPplusRuns, maxIter);
        for ri = 1:numKPPplusRuns
            if ~isempty(runs{ri})
                iters = runs{ri}.iteration;
                allCost(ri, iters) = runs{ri}.best_cost;
            end
        end
        allCost(allCost > MAX_FLOAT_THRESH) = NaN;
        meanC  = mean(allCost, 1, 'omitnan');
        itrVec = 1:maxIter;
        validIdx = ~isnan(meanC);
        if any(validIdx)
            plot(itrVec(validIdx), meanC(validIdx), ':', ...
                'Color', deltaColors(di, :), 'LineWidth', 1.8, ...
                'DisplayName', [deltaLabels{di} ' (KPAX+)']);
        end
    end

    % --- KPAX curves on iteration plot (dashed, same delta colors) ---
    for di = 1:length(deltas)
        runs = kpaxIterData{di};
        maxIter = 0;
        for ri = 1:numKPAXRuns
            if ~isempty(runs{ri}), maxIter = max(maxIter, max(runs{ri}.iteration)); end
        end
        if maxIter == 0, continue; end
        allCost = NaN(numKPAXRuns, maxIter);
        for ri = 1:numKPAXRuns
            if ~isempty(runs{ri})
                iters = runs{ri}.iteration;
                allCost(ri, iters) = runs{ri}.best_cost;
            end
        end
        allCost(allCost > MAX_FLOAT_THRESH) = NaN;
        meanC  = mean(allCost, 1, 'omitnan');
        itrVec = 1:maxIter;
        validIdx = ~isnan(meanC);
        if any(validIdx)
            plot(itrVec(validIdx), meanC(validIdx), '--', ...
                'Color', kpaxDeltaColors(di, :), 'LineWidth', 1.8, ...
                'DisplayName', [deltaLabels{di} ' (KPAX)']);
        end
    end

    xlabel('Iteration');
    ylabel('Path Cost (workspace distance)');
    title(sprintf('Best Cost vs Iteration \x2014 %s Environment', envTitle));
    legend('Location', 'best', 'FontSize', 7);
    grid on;
    set(gca, 'FontSize', 10);

    %% ==================================================================
    %  FIGURE: Best Cost vs Elapsed Time (mean +/- std)
    %  Same style as iteration plot but with time on x-axis
    %  ==================================================================
    figNum = figNum + 1;
    figure('Name', sprintf('%s - Cost vs Time', envTitle), ...
           'Position', [100 100 900 600]);
    hold on;

    % Determine global time range across all deltas and KPAX
    globalMaxTime = 0;
    for di = 1:length(deltas)
        runs = iterData{di};
        for ri = 1:numRuns
            if ~isempty(runs{ri})
                globalMaxTime = max(globalMaxTime, max(runs{ri}.elapsed_time_ms));
            end
        end
    end
    for di = 1:length(deltas)
        runs = kpaxIterData{di};
        for ri = 1:numKPAXRuns
            if ~isempty(runs{ri})
                globalMaxTime = max(globalMaxTime, max(runs{ri}.elapsed_time_ms));
            end
        end
    end
    for di = 1:length(deltas)
        runs = kpaxPlusIterData{di};
        for ri = 1:numKPPplusRuns
            if ~isempty(runs{ri})
                globalMaxTime = max(globalMaxTime, max(runs{ri}.elapsed_time_ms));
            end
        end
    end

    if globalMaxTime > 0
        % Common time grid for interpolation
        numTimeSamples = 500;
        commonTime = linspace(0, globalMaxTime, numTimeSamples);

        for di = 1:length(deltas)
            runs = iterData{di};

            allCostTime = NaN(numRuns, numTimeSamples);
            for ri = 1:numRuns
                if ~isempty(runs{ri})
                    riCost = runs{ri}.best_cost;
                    riCost(riCost > MAX_FLOAT_THRESH) = NaN;
                    riTime = runs{ri}.elapsed_time_ms;
                    
                    % Handle duplicate time values - keep last occurrence
                    [riTime, uniqueIdx] = unique(riTime, 'last');
                    riCost = riCost(uniqueIdx);
                    
                    if length(riTime) >= 2
                        allCostTime(ri, :) = interp1(riTime, riCost, commonTime, 'previous', NaN);
                    end
                end
            end

            meanC  = mean(allCostTime, 1, 'omitnan');
            stdC   = std(allCostTime, 0, 1, 'omitnan');

            validIdx = ~isnan(meanC);
            if any(validIdx)
                xFill = [commonTime(validIdx), fliplr(commonTime(validIdx))];
                yFill = [meanC(validIdx) + stdC(validIdx), fliplr(meanC(validIdx) - stdC(validIdx))];
                fill(xFill, yFill, deltaColors(di, :), ...
                    'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
                plot(commonTime(validIdx), meanC(validIdx), deltaStyles{di}, ...
                    'Color', deltaColors(di, :), 'LineWidth', 1.8, ...
                    'DisplayName', deltaLabels{di});
            end
        end

        % --- KPAXPlus curves (dotted, same delta colors) ---
        for di = 1:length(deltas)
            runs = kpaxPlusIterData{di};
            allCostTime = NaN(numKPPplusRuns, numTimeSamples);
            for ri = 1:numKPPplusRuns
                if ~isempty(runs{ri})
                    riCost = runs{ri}.best_cost;
                    riCost(riCost > MAX_FLOAT_THRESH) = NaN;
                    riTime = runs{ri}.elapsed_time_ms;
                    [riTime, uniqueIdx] = unique(riTime, 'last');
                    riCost = riCost(uniqueIdx);
                    if length(riTime) >= 2
                        allCostTime(ri, :) = interp1(riTime, riCost, commonTime, 'previous', NaN);
                    end
                end
            end
            meanC = mean(allCostTime, 1, 'omitnan');
            validIdx = ~isnan(meanC);
            if any(validIdx)
                plot(commonTime(validIdx), meanC(validIdx), ':', ...
                    'Color', deltaColors(di, :), 'LineWidth', 1.8, ...
                    'DisplayName', [deltaLabels{di} ' (KPAX+)']);
            end
        end

        % --- KPAX curves (dashed, same delta colors) ---
        for di = 1:length(deltas)
            runs = kpaxIterData{di};
            allCostTime = NaN(numKPAXRuns, numTimeSamples);
            for ri = 1:numKPAXRuns
                if ~isempty(runs{ri})
                    riCost = runs{ri}.best_cost;
                    riCost(riCost > MAX_FLOAT_THRESH) = NaN;
                    riTime = runs{ri}.elapsed_time_ms;
                    [riTime, uniqueIdx] = unique(riTime, 'last');
                    riCost = riCost(uniqueIdx);
                    if length(riTime) >= 2
                        allCostTime(ri, :) = interp1(riTime, riCost, commonTime, 'previous', NaN);
                    end
                end
            end
            meanC = mean(allCostTime, 1, 'omitnan');
            validIdx = ~isnan(meanC);
            if any(validIdx)
                plot(commonTime(validIdx), meanC(validIdx), '--', ...
                    'Color', kpaxDeltaColors(di, :), 'LineWidth', 1.8, ...
                    'DisplayName', [deltaLabels{di} ' (KPAX)']);
            end
        end
    end

    xlabel('Elapsed Time (ms)');
    ylabel('Path Cost (workspace distance)');
    title(sprintf('Best Cost vs Time \x2014 %s Environment', envTitle));
    legend('Location', 'best', 'FontSize', 7);
    xlim([0 1000]);  % 10 seconds max
    grid on;
    set(gca, 'FontSize', 10);

    %% ==================================================================
    %  FIGURE: Best Cost vs Time — LINE PLOT (mean + individual runs)
    %  Kept alongside box plot for reference
    %  ==================================================================
    figNum = figNum + 1;
    figure('Name', sprintf('%s - Cost vs Time (Lines)', envTitle), ...
           'Position', [120 120 900 600]);
    hold on;

    for di = 1:length(deltas)
        runs = iterData{di};

        % Thin individual runs
        for ri = 1:numRuns
            if ~isempty(runs{ri})
                costs = runs{ri}.best_cost;
                costs(costs > MAX_FLOAT_THRESH) = NaN;
                t = runs{ri}.elapsed_time_ms;
                plot(t, costs, '-', 'Color', [deltaColors(di,:), 0.2], ...
                    'LineWidth', 0.5, 'HandleVisibility', 'off');
            end
        end

        % Bold mean
        if ~isempty(runs{1})
            refTime = runs{1}.elapsed_time_ms;
            refCost = runs{1}.best_cost;
            refCost(refCost > MAX_FLOAT_THRESH) = NaN;
            
            % Handle duplicates in reference
            [refTime, uniqueIdx] = unique(refTime, 'last');
            refCost = refCost(uniqueIdx);
            
            allCostTime = NaN(numRuns, length(refTime));
            allCostTime(1, :) = refCost;
            for ri = 2:numRuns
                if ~isempty(runs{ri})
                    riCost = runs{ri}.best_cost;
                    riCost(riCost > MAX_FLOAT_THRESH) = NaN;
                    riTime = runs{ri}.elapsed_time_ms;
                    
                    % Handle duplicate time values
                    [riTime, uniqueIdx] = unique(riTime, 'last');
                    riCost = riCost(uniqueIdx);
                    
                    if length(riTime) >= 2
                        allCostTime(ri, :) = interp1(riTime, riCost, ...
                            refTime, 'previous', NaN);
                    end
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

    % --- KPAX curves: all runs (thin) + per-delta mean (dashed, KPAX colors) ---
    for di = 1:length(deltas)
        runs = kpaxIterData{di};

        % Thin individual runs
        for ri = 1:numKPAXRuns
            if ~isempty(runs{ri})
                costs = runs{ri}.best_cost;
                costs(costs > MAX_FLOAT_THRESH) = NaN;
                t = runs{ri}.elapsed_time_ms;
                plot(t, costs, '-', 'Color', [kpaxDeltaColors(di,:), 0.2], ...
                    'LineWidth', 0.5, 'HandleVisibility', 'off');
            end
        end

        % Bold mean
        if isempty(runs{1}), continue; end
        refTime = runs{1}.elapsed_time_ms;
        refCost = runs{1}.best_cost;
        refCost(refCost > MAX_FLOAT_THRESH) = NaN;
        [refTime, uniqueIdx] = unique(refTime, 'last');
        refCost = refCost(uniqueIdx);
        allCostTime = NaN(numKPAXRuns, length(refTime));
        allCostTime(1, :) = refCost;
        for ri = 2:numKPAXRuns
            if ~isempty(runs{ri})
                riCost = runs{ri}.best_cost;
                riCost(riCost > MAX_FLOAT_THRESH) = NaN;
                riTime = runs{ri}.elapsed_time_ms;
                [riTime, uniqueIdx] = unique(riTime, 'last');
                riCost = riCost(uniqueIdx);
                if length(riTime) >= 2
                    allCostTime(ri, :) = interp1(riTime, riCost, refTime, 'previous', NaN);
                end
            end
        end
        meanCT = mean(allCostTime, 1, 'omitnan');
        validIdx = ~isnan(meanCT);
        if any(validIdx)
            plot(refTime(validIdx), meanCT(validIdx), '--', ...
                'Color', kpaxDeltaColors(di,:), 'LineWidth', 2.0, ...
                'DisplayName', [deltaLabels{di} ' (KPAX)']);
        end
    end

    xlabel('Elapsed Time (ms)');
    ylabel('Path Cost (workspace distance)');
    title(sprintf('Cost Convergence over Wall Time \x2014 %s Environment', envTitle));
    legend('Location', 'best', 'FontSize', 7);
    xlim([0 1000]);  % 10 seconds max
    grid on;
    set(gca, 'FontSize', 10);

    %% ==================================================================
    %  FIGURE: Summary Bar Charts
    %  ==================================================================
    if ~isempty(summaryTable)
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Summary', envTitle), ...
               'Position', [150 150 1400 450]);

        nDelta = length(deltas);

        % --- First Solution Iteration ---
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

        % --- Final Best Cost ---
        subplot(1, 3, 2); hold on;

        barData2 = NaN(1, nDelta);
        barErr2  = NaN(1, nDelta);

        for di = 1:nDelta
            mask = strcmp(summaryTable.delta_label, deltas{di}) & ...
                   strcmp(summaryTable.environment, env);
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

        % --- Total Execution Time ---
        subplot(1, 3, 3); hold on;

        barData3 = NaN(1, nDelta);
        barErr3  = NaN(1, nDelta);

        for di = 1:nDelta
            mask = strcmp(summaryTable.delta_label, deltas{di}) & ...
                   strcmp(summaryTable.environment, env);
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

        sgtitle(sprintf('KinoPaxPlus Delta Comparison \x2014 %s (mean \\pm std over %d runs)', ...
            envTitle, numRuns), 'FontSize', 12, 'FontWeight', 'bold');
    end

    %% ==================================================================
    %  FIGURE: Tree Size Growth vs Iteration
    %  ==================================================================
    figNum = figNum + 1;
    figure('Name', sprintf('%s - Tree Size', envTitle), ...
           'Position', [200 50 900 600]);
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
    title(sprintf('Tree Size Growth over Iterations \x2014 %s Environment', envTitle));
    legend('Location', 'best', 'FontSize', 7);
    grid on;
    set(gca, 'FontSize', 10);

    %% ==================================================================
    %  FIGURE: Solution Success Rate
    %  ==================================================================
    figNum = figNum + 1;
    figure('Name', sprintf('%s - Success Rate', envTitle), ...
           'Position', [250 100 700 400]);

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
    title(sprintf('Solution Success Rate by Delta \x2014 %s', envTitle), 'FontSize', 13);
    grid on;

    % Add percentage labels on bars
    xtips = b4.XEndPoints;
    ytips = b4.YEndPoints;
    labels = string(round(successRate)) + "%";
    text(xtips, ytips + 2, labels, 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');

    %% ==================================================================
    %  FIGURE: Final Best Cost — KinoPaxPlus vs KPAXPlus (per delta)
    %  Computed from per-iteration data (robust to the summary-table
    %  delta_label limitation, where all KPAXPlus rows share "KPAXPlus").
    %  ==================================================================
    figNum = figNum + 1;
    figure('Name', sprintf('%s - KinoPaxPlus vs KPAXPlus Final Cost', envTitle), ...
           'Position', [300 150 900 500]);
    nDelta = length(deltas);
    kppFinal      = NaN(1, nDelta);
    kpaxplusFinal = NaN(1, nDelta);
    for di = 1:nDelta
        kppFinal(di)      = meanFinalCost(iterData{di},         numRuns,        MAX_FLOAT_THRESH);
        kpaxplusFinal(di) = meanFinalCost(kpaxPlusIterData{di}, numKPPplusRuns, MAX_FLOAT_THRESH);
    end
    bar([kppFinal(:), kpaxplusFinal(:)]);
    legend({'KinoPaxPlus', 'KPAXPlus'}, 'Location', 'best', 'FontSize', 8);
    set(gca, 'XTick', 1:nDelta, 'XTickLabel', deltaLabels, 'FontSize', 8);
    xtickangle(25);
    ylabel('Final Path Cost (workspace distance)');
    title(sprintf('Final Best Cost: KinoPaxPlus vs KPAXPlus \x2014 %s', envTitle));
    grid on;

end  % environment loop

fprintf('\nAll figures generated (%d total).\n', figNum);

%% --- Helper: find first solution iteration ---
function firstIter = firstSolutionIter(tbl, thresh)
    solIdx = find(tbl.best_cost < thresh, 1, 'first');
    if isempty(solIdx)
        firstIter = -1;
    else
        firstIter = tbl.iteration(solIdx);
    end
end

%% --- Helper: find first solution time (ms) ---
function firstTime = firstSolutionTime(tbl, thresh)
    solIdx = find(tbl.best_cost < thresh, 1, 'first');
    if isempty(solIdx)
        firstTime = NaN;
    else
        firstTime = tbl.elapsed_time_ms(solIdx);
    end
end

%% --- Helper: mean final best cost across a delta's runs ---
function c = meanFinalCost(runs, numRuns, thresh)
    vals = NaN(1, numRuns);
    for ri = 1:numel(runs)
        if ~isempty(runs{ri})
            costs = runs{ri}.best_cost;
            costs(costs > thresh) = NaN;
            v = costs(~isnan(costs));
            if ~isempty(v), vals(ri) = v(end); end
        end
    end
    c = mean(vals, 'omitnan');
end
