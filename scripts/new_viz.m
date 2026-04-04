%% KinoPaxPlus Delta Benchmark - Cost vs Time Only
% Simplified version that only generates cost vs time plots.
%
% Output directory: Data/Benchmarks/KinoPaxPlusDelta/
%   KinoPaxPlus:  {env}_delta{label}_run{n}.csv
%   KPAX:         {env}_KPAX_run{n}.csv
%
% Per-iteration columns: iteration, frontier_size, tree_size, elapsed_time_ms, best_cost

clear; clc; close all;

%% --- Configuration ---
dataDir = 'KinoPaxPlusDelta500kmore';

environments = {'trees', 'house'};
envTitles    = {'Trees', 'House'};

deltas      = {'extra_large', 'larger', 'large','othersize', 'otherothersize', 'med_large', 'med_small'};
deltaLabels = {'Extra-Large-\delta (3.3k)', ...
                'Larger-\delta (14k)', ...
                'Large-\delta (27k)', ...
                'OtherSize (46k)', ...
                'YetAnotherSize (110k)', ...
               'Med-Large (216k)', ...
               'Med-Small (1.7M)'};

deltaColors = [0.2 0.4 0.8;    % extra_large - blue
               0.9 0.5 0.1;    % larger      - orange
               0.2 0.7 0.3;    % large       - green
               0.4 0.4 0.3;    % othersize   - olive
               0.4 0.2 0.8;    % otherothersize - purple
               0.8 0.2 0.2;    % med_large   - red
               0.6 0.1 0.7];   % med_small   - purple

deltaStyles = {'-', '--', '-.', ':', '-.', ':', '-'};

kpaxColor = [0.1 0.1 0.1];     % near-black for KPAX curve
kpaxFirstColor = [0.5 0.5 0.5]; % gray for reference lines
kpaxFinalColor = [0.2 0.2 0.2]; % dark gray
kpaxTimeColor  = [0.6 0.3 0.6]; % purple-ish

numRuns      = 50;   % KinoPaxPlus runs per delta
numKPAXRuns  = 50;   % KPAX baseline runs

% Model 1 MAX_FLOAT is 1e38
MAX_FLOAT_THRESH = 1e30;

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

    %% --- Load KPAX Baseline Data ---
    kpaxData = cell(1, numKPAXRuns);
    for ri = 0:(numKPAXRuns - 1)
        fname = sprintf('%s_KPAX_run%d.csv', env, ri);
        fpath = fullfile(dataDir, fname);
        if isfile(fpath)
            kpaxData{ri + 1} = readtable(fpath);
        end
    end
    fprintf('  KPAX baseline data loaded (%d runs).\n', numKPAXRuns);

    %% --- Compute KPAX Summary Statistics ---
    kpaxFirstCosts = NaN(1, numKPAXRuns);
    kpaxFinalCosts = NaN(1, numKPAXRuns);
    kpaxFirstTimes = NaN(1, numKPAXRuns);

    for ri = 1:numKPAXRuns
        if ~isempty(kpaxData{ri})
            tbl = kpaxData{ri};
            costs = tbl.best_cost;
            costs(costs > MAX_FLOAT_THRESH) = NaN;

            % First solution cost & time
            solIdx = find(tbl.best_cost < MAX_FLOAT_THRESH, 1, 'first');
            if ~isempty(solIdx)
                kpaxFirstCosts(ri) = tbl.best_cost(solIdx);
                kpaxFirstTimes(ri) = tbl.elapsed_time_ms(solIdx);
            end

            % Final best cost (last valid value)
            validCosts = costs(~isnan(costs));
            if ~isempty(validCosts)
                kpaxFinalCosts(ri) = validCosts(end);
            end
        end
    end

    kpaxMeanFirstCost = mean(kpaxFirstCosts, 'omitnan');
    kpaxMeanFinalCost = mean(kpaxFinalCosts, 'omitnan');
    kpaxMeanFirstTime = mean(kpaxFirstTimes, 'omitnan');

    fprintf('  KPAX mean first cost:  %.4f\n', kpaxMeanFirstCost);
    fprintf('  KPAX mean final cost:  %.4f\n', kpaxMeanFinalCost);
    fprintf('  KPAX mean first time:  %.1f ms\n', kpaxMeanFirstTime);

    %% ==================================================================
    %  FIGURE 1: Best Cost vs Elapsed Time (mean +/- std)
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
    for ri = 1:numKPAXRuns
        if ~isempty(kpaxData{ri})
            globalMaxTime = max(globalMaxTime, max(kpaxData{ri}.elapsed_time_ms));
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

        % --- KPAX curve ---
        kpaxAllCostTime = NaN(numKPAXRuns, numTimeSamples);
        for ri = 1:numKPAXRuns
            if ~isempty(kpaxData{ri})
                riCost = kpaxData{ri}.best_cost;
                riCost(riCost > MAX_FLOAT_THRESH) = NaN;
                riTime = kpaxData{ri}.elapsed_time_ms;
                
                % Handle duplicate time values - keep last occurrence
                [riTime, uniqueIdx] = unique(riTime, 'last');
                riCost = riCost(uniqueIdx);
                
                if length(riTime) >= 2
                    kpaxAllCostTime(ri, :) = interp1(riTime, riCost, commonTime, 'previous', NaN);
                end
            end
        end

        kpaxMeanC = mean(kpaxAllCostTime, 1, 'omitnan');
        kpaxStdC  = std(kpaxAllCostTime, 0, 1, 'omitnan');
        kpaxValidIdx = ~isnan(kpaxMeanC);

        if any(kpaxValidIdx)
            xFill = [commonTime(kpaxValidIdx), fliplr(commonTime(kpaxValidIdx))];
            yFill = [kpaxMeanC(kpaxValidIdx) + kpaxStdC(kpaxValidIdx), ...
                     fliplr(kpaxMeanC(kpaxValidIdx) - kpaxStdC(kpaxValidIdx))];
            fill(xFill, yFill, kpaxColor, ...
                'FaceAlpha', 0.10, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            plot(commonTime(kpaxValidIdx), kpaxMeanC(kpaxValidIdx), '-', ...
                'Color', kpaxColor, 'LineWidth', 2.0, ...
                'DisplayName', 'KPAX');
        end

        % KPAX reference lines
        if ~isnan(kpaxMeanFirstCost)
            yline(kpaxMeanFirstCost, '--', 'Color', kpaxFirstColor, 'LineWidth', 1.2, ...
                'Label', 'KPAX first cost', 'HandleVisibility', 'off');
        end
        if ~isnan(kpaxMeanFinalCost)
            yline(kpaxMeanFinalCost, '-.', 'Color', kpaxFinalColor, 'LineWidth', 1.2, ...
                'Label', 'KPAX final cost', 'HandleVisibility', 'off');
        end
        if ~isnan(kpaxMeanFirstTime)
            xline(kpaxMeanFirstTime, '--', 'Color', kpaxTimeColor, 'LineWidth', 1.2, ...
                'Label', sprintf('KPAX t_{first} = %.0f ms', kpaxMeanFirstTime), ...
                'HandleVisibility', 'off');
        end
    end

    xlabel('Elapsed Time (ms)');
    ylabel('Path Cost (workspace distance)');
    title(sprintf('Best Cost vs Time \x2014 %s Environment', envTitle));
    legend('Location', 'best', 'FontSize', 7);
    xlim([0 1000]);  % 1 second max
    grid on;
    set(gca, 'FontSize', 10);

    %% ==================================================================
    %  FIGURE 2: Best Cost vs Time — LINE PLOT (mean + individual runs)
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

    % --- KPAX curve: thin individual + bold mean ---
    for ri = 1:numKPAXRuns
        if ~isempty(kpaxData{ri})
            costs = kpaxData{ri}.best_cost;
            costs(costs > MAX_FLOAT_THRESH) = NaN;
            t = kpaxData{ri}.elapsed_time_ms;
            plot(t, costs, '-', 'Color', [kpaxColor, 1.0], ...
                'LineWidth', 0.5, 'HandleVisibility', 'off');
        end
    end
    % KPAX mean
    if ~isempty(kpaxData{1})
        refTime = kpaxData{1}.elapsed_time_ms;
        refCost = kpaxData{1}.best_cost;
        refCost(refCost > MAX_FLOAT_THRESH) = NaN;
        
        % Handle duplicates in reference
        [refTime, uniqueIdx] = unique(refTime, 'last');
        refCost = refCost(uniqueIdx);
        
        kpaxAllCostTime = NaN(numKPAXRuns, length(refTime));
        kpaxAllCostTime(1, :) = refCost;
        for ri = 2:numKPAXRuns
            if ~isempty(kpaxData{ri})
                riCost = kpaxData{ri}.best_cost;
                riCost(riCost > MAX_FLOAT_THRESH) = NaN;
                riTime = kpaxData{ri}.elapsed_time_ms;
                
                % Handle duplicate time values
                [riTime, uniqueIdx] = unique(riTime, 'last');
                riCost = riCost(uniqueIdx);
                
                if length(riTime) >= 2
                    kpaxAllCostTime(ri, :) = interp1(riTime, riCost, ...
                        refTime, 'previous', NaN);
                end
            end
        end
        kpaxMeanCT = mean(kpaxAllCostTime, 1, 'omitnan');
        validIdx = ~isnan(kpaxMeanCT);
        if any(validIdx)
            plot(refTime(validIdx), kpaxMeanCT(validIdx), '-', ...
                'Color', kpaxColor, 'LineWidth', 2.0, ...
                'DisplayName', 'KPAX');
        end
    end

    % KPAX reference lines
    if ~isnan(kpaxMeanFirstCost)
        yline(kpaxMeanFirstCost, '--', 'Color', kpaxFirstColor, 'LineWidth', 1.5, ...
            'Label', 'KPAX first cost', 'DisplayName', 'KPAX First Cost');
    end
    if ~isnan(kpaxMeanFinalCost)
        yline(kpaxMeanFinalCost, '-.', 'Color', kpaxFinalColor, 'LineWidth', 1.5, ...
            'Label', 'KPAX final cost', 'DisplayName', 'KPAX Final Cost');
    end
    if ~isnan(kpaxMeanFirstTime)
        xline(kpaxMeanFirstTime, '--', 'Color', kpaxTimeColor, 'LineWidth', 1.5, ...
            'Label', sprintf('KPAX t_{first} = %.0f ms', kpaxMeanFirstTime), ...
            'DisplayName', 'KPAX First Sol Time');
    end

    xlabel('Elapsed Time (ms)');
    ylabel('Path Cost (workspace distance)');
    title(sprintf('Cost Convergence over Wall Time \x2014 %s Environment', envTitle));
    legend('Location', 'best', 'FontSize', 7);
    xlim([0 1000]);  % 1 second max
    grid on;
    set(gca, 'FontSize', 10);

end  % environment loop

fprintf('\nAll figures generated (%d total).\n', figNum);