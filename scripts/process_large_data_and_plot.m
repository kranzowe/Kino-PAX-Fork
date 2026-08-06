%% Large-Delta Benchmark Visualization — all planners, single delta
% Reads per-iteration CSVs produced by kinopaxstar_large_benchmark.cu
% (run via run_large_benchmark.sh) and compares all seven planners at the
% single Large delta (R1 = 27k regions). Single-delta simplification of
% process_delta_data_and_plot.m: the per-delta tiled panels collapse to one
% axes each, and the cross-delta grouped bars are re-keyed to planner-on-x-axis.
%
% Per-run CSVs (in dataDir):
%   KinoPaxPlus:                     {env}_delta{label}_run{n}.csv
%   KPAX:                            {env}_KPAX_delta{label}_run{n}.csv
%   PruneKPAX:                       {env}_PruneKPAX_delta{label}_run{n}.csv
%   KinoPaxSTAR:                     {env}_KinoPaxSTAR_delta{label}_run{n}.csv
%   KinoPaxSTARcostprune:            {env}_KinoPaxSTARcostprune_delta{label}_run{n}.csv
%   KinoPaxSTARNoPrune:              {env}_KinoPaxSTARNoPrune_delta{label}_run{n}.csv
%   KinoPaxSTARNoPruneNoSpatialHash: {env}_KinoPaxSTARNoPruneNoSpatialHash_delta{label}_run{n}.csv
%
% Per-iteration columns: iteration, frontier_size, tree_size, elapsed_time_ms, best_cost
%
% FAIR-COMPARISON NOTE: an "iteration" is a different unit of work per planner
% (frontier size x branching differ), so cost-vs-TIME is the fair cross-planner
% axis; read cost-vs-iteration as within-planner. The KinoPaxSTARNoPrune vs
% KinoPaxSTARNoPruneNoSpatialHash pair should track on cost (same search) and
% differ mainly on the TIME axis (spatial-hash speedup).

clear; clc; close all;

%% --- Configuration ---
dataDir = '';   % results folder ('' = current directory)

environments = {'house'};
envTitles    = {'House'};

% Single delta (Large): W_R1=10 C_R1=1 V_R1=3 -> 10^3 * 3^3 = 27k regions
deltas      = {'large'};
deltaLabels = {'Large-\delta (27k)'};

% Planners overlaid in every panel (color = planner). The two NoPrune variants
% are adjacent so the spatial-hash on/off pair is easy to read.
plannerNames  = {'KinoPaxPlus', 'KinoPaxSTAR', 'KinoPaxSTARcostprune', ...
                 'KinoPaxSTARNoPrune', 'KinoPaxSTARNoPruneNoSpatialHash', 'KPAX', 'PruneKPAX'};
plannerColors = [0.20 0.40 0.80;    % KinoPaxPlus                     - blue
                 0.55 0.15 0.60;    % KinoPaxSTAR                     - purple
                 0.10 0.60 0.55;    % KinoPaxSTARcostprune            - teal
                 0.20 0.65 0.25;    % KinoPaxSTARNoPrune              - green
                 0.80 0.20 0.20;    % KinoPaxSTARNoPruneNoSpatialHash - red
                 0.10 0.10 0.10;    % KPAX                            - near-black
                 0.85 0.45 0.10];   % PruneKPAX                       - orange
plannerStyles = {'-', '-', '-', '-', '-', '-', '-'};
numRunsPer    = [50, 50, 50, 50, 50, 50, 50];   % max runs searched per planner (missing files skipped)

MAX_FLOAT_THRESH = 1e30;   % best_cost sentinel (MAX_FLOAT / INFINITY) -> NaN
numTimeSamples   = 500;

nDelta   = numel(deltas);   % == 1 here
nPlanner = numel(plannerNames);
di       = 1;               % single delta index used throughout

%% ======================================================================
figNum = 0;
for ei = 1:numel(environments)
    env      = environments{ei};
    envTitle = envTitles{ei};
    fprintf('\n=== Environment: %s ===\n', env);

    % --- Load every planner's runs for the single delta: R{pi} = cell of tables ---
    R = cell(1, nPlanner);
    for pi = 1:nPlanner
        R{pi} = loadRuns(dataDir, env, plannerNames{pi}, deltas{di}, numRunsPer(pi));
        fprintf('  %-32s delta=%-9s : %d runs\n', plannerNames{pi}, deltas{di}, numel(R{pi}));
    end

    %% ---------- FIGURE 1: Best Cost vs Iteration (single panel) ----------
    figNum = figNum + 1;
    figure('Name', sprintf('%s - Cost vs Iteration', envTitle), 'Position', [40 40 820 620]);
    hold on;
    for pi = 1:nPlanner
        plotBandIter(R{pi}, 'best_cost', plannerColors(pi, :), plannerStyles{pi}, plannerNames{pi});
    end
    xlabel('Iteration'); ylabel('Path Cost (workspace distance)'); grid on;
    legend('Location', 'best', 'FontSize', 8);
    title(sprintf('Best Cost vs Iteration \x2014 %s, %s', envTitle, deltaLabels{di}), 'FontWeight', 'bold');

    %% ---------- FIGURE 2: Best Cost vs Time (fair axis, single panel) ----------
    figNum = figNum + 1;
    figure('Name', sprintf('%s - Cost vs Time', envTitle), 'Position', [70 60 820 620]);
    hold on;
    tmax = globalMaxTime(R);
    if tmax > 0
        ct = linspace(0, tmax, numTimeSamples);
        for pi = 1:nPlanner
            plotBandTime(R{pi}, 'best_cost', ct, plannerColors(pi, :), plannerStyles{pi}, plannerNames{pi});
        end
    end
    xlabel('Elapsed Time (ms)'); ylabel('Path Cost (workspace distance)'); grid on;
    legend('Location', 'best', 'FontSize', 8);
    title(sprintf('Best Cost vs Time \x2014 %s, %s', envTitle, deltaLabels{di}), 'FontWeight', 'bold');

    %% ---------- FIGURE 3: Tree Size vs Iteration (single panel) ----------
    figNum = figNum + 1;
    figure('Name', sprintf('%s - Tree Size', envTitle), 'Position', [100 40 820 620]);
    hold on;
    for pi = 1:nPlanner
        plotBandIter(R{pi}, 'tree_size', plannerColors(pi, :), plannerStyles{pi}, plannerNames{pi});
    end
    xlabel('Iteration'); ylabel('Tree Size'); grid on;
    legend('Location', 'best', 'FontSize', 8);
    title(sprintf('Tree Size Growth vs Iteration \x2014 %s, %s', envTitle, deltaLabels{di}), 'FontWeight', 'bold');

    %% ---------- Aggregate summary metrics per planner (single delta) ----------
    mFirstIter = NaN(1, nPlanner); eFirstIter = NaN(1, nPlanner);
    mFinalCost = NaN(1, nPlanner); eFinalCost = NaN(1, nPlanner);
    mTotalTime = NaN(1, nPlanner); eTotalTime = NaN(1, nPlanner);
    mSuccess   = NaN(1, nPlanner);
    for pi = 1:nPlanner
        runs = R{pi};
        fiVals = []; fcVals = []; ttVals = [];
        nSol = 0; nTot = 0;
        for ri = 1:numel(runs)
            if isempty(runs{ri}), continue; end
            nTot = nTot + 1;
            fi = firstSolIter(runs{ri}, MAX_FLOAT_THRESH);
            if fi > 0, nSol = nSol + 1; fiVals(end + 1) = fi; end %#ok<AGROW>
            fc = finalCost(runs{ri}, MAX_FLOAT_THRESH);
            if ~isnan(fc), fcVals(end + 1) = fc; end %#ok<AGROW>
            ttVals(end + 1) = runs{ri}.elapsed_time_ms(end) / 1000; %#ok<AGROW>
        end
        if ~isempty(fiVals), mFirstIter(pi) = mean(fiVals); eFirstIter(pi) = std(fiVals); end
        if ~isempty(fcVals), mFinalCost(pi) = mean(fcVals); eFinalCost(pi) = std(fcVals); end
        if ~isempty(ttVals), mTotalTime(pi) = mean(ttVals); eTotalTime(pi) = std(ttVals); end
        if nTot > 0,         mSuccess(pi)   = 100 * nSol / nTot; end
    end

    %% ---------- FIGURE 4: Summary bars (one bar per planner) ----------
    figNum = figNum + 1;
    figure('Name', sprintf('%s - Summary', envTitle), 'Position', [130 120 1500 460]);

    subplot(1, 3, 1);
    plannerBar(mFirstIter, eFirstIter, plannerNames, plannerColors, 'Iteration', 'First Solution Iteration');

    subplot(1, 3, 2);
    plannerBar(mFinalCost, eFinalCost, plannerNames, plannerColors, 'Path Cost (workspace distance)', 'Final Best Cost');

    subplot(1, 3, 3);
    plannerBar(mTotalTime, eTotalTime, plannerNames, plannerColors, 'Time (s)', 'Total Execution Time');

    sgtitle(sprintf('Planner Comparison at %s \x2014 %s (mean \\pm std)', deltaLabels{di}, envTitle), ...
        'FontSize', 12, 'FontWeight', 'bold');

    %% ---------- FIGURE 5: Solution Success Rate (one bar per planner) ----------
    figNum = figNum + 1;
    figure('Name', sprintf('%s - Success Rate', envTitle), 'Position', [160 160 900 480]);
    plannerBar(mSuccess, [], plannerNames, plannerColors, 'Success Rate (%)', ...
        sprintf('Solution Success Rate \x2014 %s, %s', envTitle, deltaLabels{di}));
    ylim([0 110]);

end  % environment loop

fprintf('\nAll figures generated (%d total).\n', figNum);

%% ====================== helper functions ======================

function runs = loadRuns(dataDir, env, planner, delta, numRuns)
    % Load a planner's per-run CSVs for one delta; missing files are skipped.
    runs = {};
    for ri = 0:(numRuns - 1)
        switch planner
            case 'KinoPaxPlus'
                fn = sprintf('%s_delta%s_run%d.csv', env, delta, ri);
            case 'KPAX'
                fn = sprintf('%s_KPAX_delta%s_run%d.csv', env, delta, ri);
            case 'PruneKPAX'
                fn = sprintf('%s_PruneKPAX_delta%s_run%d.csv', env, delta, ri);
            case 'KinoPaxSTAR'
                fn = sprintf('%s_KinoPaxSTAR_delta%s_run%d.csv', env, delta, ri);
            case 'KinoPaxSTARcostprune'
                fn = sprintf('%s_KinoPaxSTARcostprune_delta%s_run%d.csv', env, delta, ri);
            case 'KinoPaxSTARNoPrune'
                fn = sprintf('%s_KinoPaxSTARNoPrune_delta%s_run%d.csv', env, delta, ri);
            case 'KinoPaxSTARNoPruneNoSpatialHash'
                fn = sprintf('%s_KinoPaxSTARNoPruneNoSpatialHash_delta%s_run%d.csv', env, delta, ri);
            otherwise
                error('unknown planner %s', planner);
        end
        fp = fullfile(dataDir, fn);
        if isfile(fp)
            runs{end + 1} = readtable(fp); %#ok<AGROW>
        end
    end
end

function tmax = globalMaxTime(runsCell)
    % Max elapsed_time_ms across a cell array of run-cell-arrays (all planners).
    tmax = 0;
    for g = 1:numel(runsCell)
        runs = runsCell{g};
        for ri = 1:numel(runs)
            if ~isempty(runs{ri})
                tmax = max(tmax, max(runs{ri}.elapsed_time_ms));
            end
        end
    end
end

function plotBandIter(runs, col, color, style, name)
    % mean +/- std of column 'col' vs iteration across runs
    if isempty(runs), return; end
    maxIter = 0;
    for ri = 1:numel(runs)
        if ~isempty(runs{ri}), maxIter = max(maxIter, max(runs{ri}.iteration)); end
    end
    if maxIter == 0, return; end
    A = NaN(numel(runs), maxIter);
    for ri = 1:numel(runs)
        if isempty(runs{ri}), continue; end
        it = runs{ri}.iteration;
        v  = getCol(runs{ri}, col);
        if isempty(v), continue; end
        A(ri, it) = v;
    end
    A = sanitize(A);
    drawBand(1:maxIter, mean(A, 1, 'omitnan'), std(A, 0, 1, 'omitnan'), color, style, name);
end

function plotBandTime(runs, col, commonTime, color, style, name)
    % mean +/- std of column 'col' vs a shared time grid (previous-sample hold)
    if isempty(runs), return; end
    A = NaN(numel(runs), numel(commonTime));
    for ri = 1:numel(runs)
        if isempty(runs{ri}), continue; end
        t = runs{ri}.elapsed_time_ms;
        v = getCol(runs{ri}, col);
        if isempty(v), continue; end
        [t, uix] = unique(t, 'last');
        v = v(uix);
        if numel(t) >= 2
            A(ri, :) = interp1(t, v, commonTime, 'previous', NaN);
        end
    end
    A = sanitize(A);
    drawBand(commonTime, mean(A, 1, 'omitnan'), std(A, 0, 1, 'omitnan'), color, style, name);
end

function v = getCol(tbl, col)
    if any(strcmp(col, tbl.Properties.VariableNames))
        v = tbl.(col);
    else
        v = [];
    end
end

function A = sanitize(A)
    % Drop cost sentinels (MAX_FLOAT / INFINITY) so they don't distort means.
    A(A > 1e30) = NaN;
    A(A == -1)  = NaN;   % harmless for cost/tree_size; guards diagnostic sentinels
end

function drawBand(x, mu, sd, color, style, name)
    valid = ~isnan(mu);
    if ~any(valid), return; end
    xv = x(valid); mv = mu(valid); sv = sd(valid); sv(isnan(sv)) = 0;
    fill([xv, fliplr(xv)], [mv + sv, fliplr(mv - sv)], color, ...
        'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(xv, mv, style, 'Color', color, 'LineWidth', 1.8, 'DisplayName', name);
end

function plannerBar(mu, err, plannerNames, plannerColors, ylab, ttl)
    % One colored bar per planner on the x-axis (single-delta replacement for
    % the cross-delta grouped bars). mu/err are 1 x nPlanner row vectors.
    hold on;
    nP = numel(mu);
    for pi = 1:nP
        bar(pi, mu(pi), 0.7, 'FaceColor', plannerColors(pi, :), 'EdgeColor', 'none');
    end
    if ~isempty(err)
        errorbar(1:nP, mu, err, 'k.', 'LineWidth', 0.7, 'CapSize', 4);
    end
    set(gca, 'XTick', 1:nP, 'XTickLabel', plannerNames, 'FontSize', 8);
    xtickangle(25);
    xlim([0.5, nP + 0.5]);
    ylabel(ylab); title(ttl); grid on;
end

function it = firstSolIter(tbl, thresh)
    % Iteration at which this run first reached a finite (< thresh) best_cost; -1 if never.
    solIdx = find(tbl.best_cost < thresh, 1, 'first');
    if isempty(solIdx), it = -1; else, it = tbl.iteration(solIdx); end
end

function c = finalCost(tbl, thresh)
    % Last finite best_cost in one run; NaN if the run never found a solution.
    costs = tbl.best_cost;
    costs(costs > thresh) = NaN;
    v = costs(~isnan(costs));
    if isempty(v), c = NaN; else, c = v(end); end
end
