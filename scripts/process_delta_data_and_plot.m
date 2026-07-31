%% Delta Benchmark Visualization — all planners, per-delta panels
% Reads per-iteration CSVs produced by kinopaxplus_delta_benchmark.cu
% (run via run_delta_benchmark.sh) and compares all four planners across the
% delta (R1 discretization) sweep. One tiled panel per delta; every panel
% overlays KinoPaxPlus, KinoPaxSTAR, KPAX, and PruneKPAX (color = planner).
%
% Per-run CSVs (in dataDir):
%   KinoPaxPlus:  {env}_delta{label}_run{n}.csv
%   KPAX:         {env}_KPAX_delta{label}_run{n}.csv
%   PruneKPAX:    {env}_PruneKPAX_delta{label}_run{n}.csv
%   KinoPaxSTAR:  {env}_KinoPaxSTAR_delta{label}_run{n}.csv
%
% Per-iteration columns: iteration, frontier_size, tree_size, elapsed_time_ms, best_cost
%
% FAIR-COMPARISON NOTE: an "iteration" is a different unit of work per planner
% (frontier size x branching differ), so cost-vs-TIME is the fair cross-planner
% axis; read cost-vs-iteration as within-planner. Time panels auto-scale to each
% delta's data (no fixed xlim), so a planner still improving late is not hidden.

clear; clc; close all;

%% --- Configuration ---
dataDir = '';   % results folder ('' = current directory)

environments = {'house'};
envTitles    = {'House'};

deltas      = {'larger', 'large', 'med_large', 'med_small'};
deltaLabels = {'Larger-\delta (14k)', ...
               'Large-\delta (27k)', ...
               'Med-Large (216k)', ...
               'Med-Small (422k)'};

% Planners overlaid in every panel (color = planner, consistent with
% visualize_frontier_growth.m). Order controls legend/bar order.
plannerNames  = {'KinoPaxPlus', 'KinoPaxSTAR', 'KPAX', 'PruneKPAX'};
plannerColors = [0.20 0.40 0.80;    % KinoPaxPlus - blue
                 0.55 0.15 0.60;    % KinoPaxSTAR - purple
                 0.10 0.10 0.10;    % KPAX        - near-black
                 0.85 0.45 0.10];   % PruneKPAX   - orange
plannerStyles = {'-', '-', '-', '-'};
numRunsPer    = [50, 50, 50, 50];   % max runs searched per planner (missing files skipped)

MAX_FLOAT_THRESH = 1e30;   % best_cost sentinel (MAX_FLOAT / INFINITY) -> NaN
numTimeSamples   = 500;

nDelta   = numel(deltas);
nPlanner = numel(plannerNames);
nCols = 2; nRows = ceil(nDelta / nCols);

%% ======================================================================
figNum = 0;
for ei = 1:numel(environments)
    env      = environments{ei};
    envTitle = envTitles{ei};
    fprintf('\n=== Environment: %s ===\n', env);

    % --- Load every planner's runs for every delta: R{di, pi} = cell of tables ---
    R = cell(nDelta, nPlanner);
    for di = 1:nDelta
        for pi = 1:nPlanner
            R{di, pi} = loadRuns(dataDir, env, plannerNames{pi}, deltas{di}, numRunsPer(pi));
            fprintf('  %-12s delta=%-9s : %d runs\n', plannerNames{pi}, deltas{di}, numel(R{di, pi}));
        end
    end

    %% ---------- FIGURE 1: Best Cost vs Iteration ----------
    figNum = figNum + 1;
    figure('Name', sprintf('%s - Cost vs Iteration', envTitle), 'Position', [40 40 1150 780]);
    tl = tiledlayout(nRows, nCols, 'TileSpacing', 'compact', 'Padding', 'compact');
    for di = 1:nDelta
        nexttile; hold on;
        for pi = 1:nPlanner
            plotBandIter(R{di, pi}, 'best_cost', plannerColors(pi, :), plannerStyles{pi}, plannerNames{pi});
        end
        title(deltaLabels{di});
        xlabel('Iteration'); ylabel('Path Cost (workspace distance)'); grid on;
        if di == 1, legend('Location', 'best', 'FontSize', 7); end
    end
    title(tl, sprintf('Best Cost vs Iteration \x2014 %s Environment', envTitle), 'FontWeight', 'bold');

    %% ---------- FIGURE 2: Best Cost vs Time (fair axis) ----------
    figNum = figNum + 1;
    figure('Name', sprintf('%s - Cost vs Time', envTitle), 'Position', [70 60 1150 780]);
    tl = tiledlayout(nRows, nCols, 'TileSpacing', 'compact', 'Padding', 'compact');
    for di = 1:nDelta
        nexttile; hold on;
        tmax = globalMaxTime(R(di, :));
        if tmax > 0
            ct = linspace(0, tmax, numTimeSamples);
            for pi = 1:nPlanner
                plotBandTime(R{di, pi}, 'best_cost', ct, plannerColors(pi, :), plannerStyles{pi}, plannerNames{pi});
            end
        end
        title(deltaLabels{di});
        xlabel('Elapsed Time (ms)'); ylabel('Path Cost (workspace distance)'); grid on;
        if di == 1, legend('Location', 'best', 'FontSize', 7); end
    end
    title(tl, sprintf('Best Cost vs Time \x2014 %s Environment', envTitle), 'FontWeight', 'bold');

    %% ---------- FIGURE 3: Tree Size vs Iteration ----------
    figNum = figNum + 1;
    figure('Name', sprintf('%s - Tree Size', envTitle), 'Position', [100 40 1150 780]);
    tl = tiledlayout(nRows, nCols, 'TileSpacing', 'compact', 'Padding', 'compact');
    for di = 1:nDelta
        nexttile; hold on;
        for pi = 1:nPlanner
            plotBandIter(R{di, pi}, 'tree_size', plannerColors(pi, :), plannerStyles{pi}, plannerNames{pi});
        end
        title(deltaLabels{di});
        xlabel('Iteration'); ylabel('Tree Size'); grid on;
        if di == 1, legend('Location', 'best', 'FontSize', 7); end
    end
    title(tl, sprintf('Tree Size Growth vs Iteration \x2014 %s Environment', envTitle), 'FontWeight', 'bold');

    %% ---------- Aggregate summary metrics per (delta, planner) ----------
    mFirstIter = NaN(nDelta, nPlanner); eFirstIter = NaN(nDelta, nPlanner);
    mFinalCost = NaN(nDelta, nPlanner); eFinalCost = NaN(nDelta, nPlanner);
    mTotalTime = NaN(nDelta, nPlanner); eTotalTime = NaN(nDelta, nPlanner);
    mSuccess   = NaN(nDelta, nPlanner);
    for di = 1:nDelta
        for pi = 1:nPlanner
            runs = R{di, pi};
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
            if ~isempty(fiVals), mFirstIter(di, pi) = mean(fiVals); eFirstIter(di, pi) = std(fiVals); end
            if ~isempty(fcVals), mFinalCost(di, pi) = mean(fcVals); eFinalCost(di, pi) = std(fcVals); end
            if ~isempty(ttVals), mTotalTime(di, pi) = mean(ttVals); eTotalTime(di, pi) = std(ttVals); end
            if nTot > 0,         mSuccess(di, pi)   = 100 * nSol / nTot; end
        end
    end

    %% ---------- FIGURE 4: Summary bars (grouped by planner) ----------
    figNum = figNum + 1;
    figure('Name', sprintf('%s - Summary', envTitle), 'Position', [130 120 1500 460]);

    subplot(1, 3, 1);
    groupedBar(mFirstIter, eFirstIter, deltaLabels, plannerColors, 'Iteration', 'First Solution Iteration');
    legend(plannerNames, 'Location', 'best', 'FontSize', 7);

    subplot(1, 3, 2);
    groupedBar(mFinalCost, eFinalCost, deltaLabels, plannerColors, 'Path Cost (workspace distance)', 'Final Best Cost');

    subplot(1, 3, 3);
    groupedBar(mTotalTime, eTotalTime, deltaLabels, plannerColors, 'Time (s)', 'Total Execution Time');

    sgtitle(sprintf('Planner Comparison across Delta \x2014 %s (mean \\pm std)', envTitle), ...
        'FontSize', 12, 'FontWeight', 'bold');

    %% ---------- FIGURE 5: Solution Success Rate (grouped by planner) ----------
    figNum = figNum + 1;
    figure('Name', sprintf('%s - Success Rate', envTitle), 'Position', [160 160 900 480]);
    groupedBar(mSuccess, [], deltaLabels, plannerColors, 'Success Rate (%)', ...
        sprintf('Solution Success Rate \x2014 %s', envTitle));
    ylim([0 110]);
    legend(plannerNames, 'Location', 'best', 'FontSize', 8);

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
            otherwise
                error('unknown planner %s', planner);
        end
        fp = fullfile(dataDir, fn);
        if isfile(fp)
            runs{end + 1} = readtable(fp); %#ok<AGROW>
        end
    end
end

function tmax = globalMaxTime(groups)
    % Max elapsed_time_ms across a set of run-cell-arrays (one delta's planners).
    tmax = 0;
    for g = 1:numel(groups)
        runs = groups{g};
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

function groupedBar(M, E, deltaLabels, plannerColors, ylab, ttl)
    % Grouped bars: x-groups = deltas (rows of M), bars within group = planners (cols).
    hold on;
    b = bar(M);
    nP = size(M, 2);
    for pi = 1:nP
        b(pi).FaceColor = plannerColors(pi, :);
    end
    if ~isempty(E)
        for pi = 1:nP
            errorbar(b(pi).XEndPoints, M(:, pi)', E(:, pi)', 'k.', 'LineWidth', 0.7, 'CapSize', 4);
        end
    end
    set(gca, 'XTick', 1:size(M, 1), 'XTickLabel', deltaLabels, 'FontSize', 8);
    xtickangle(20);
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
