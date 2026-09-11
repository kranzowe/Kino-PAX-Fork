%% Paper Benchmark Plots v2 - fixed 4-planner comparison, 3 panels + summary table per (env, metric)
% Reads per-iteration CSVs produced by examples/gpu/paper_benchmark_v2.cu (run via
% scripts/run_paper_benchmark_v2.sh) -- countingstars_sweep.cu's proven-reliable harness, minimally
% adapted (all four environments, single fixed operating points) to reproduce
% paper_benchmark.cu's own headline comparison without paper_benchmark.cu's unresolved hang at
% `tiny`. Runs the same MODEL 2 / Dubins Airplane paper_benchmark.cu targets, but with a corrected
% region-discretization dimension breakdown (C_DIM=2 for yaw+pitch, V_DIM=1 for airspeed only,
% both properly bounded) instead of paper_benchmark.cu's own C_DIM=0/V_DIM=3 -- so results here are
% NOT directly comparable to paper_benchmark.cu's own numbers. See paper_benchmark_v2.cu /
% run_paper_benchmark_v2.sh for the full derivation. Otherwise identical in structure/presentation
% to process_paper_benchmark_and_plot.m.
%
% A FIXED COMPARISON, not a sweep -- there is no grid here, so the series list below is NOT built
% from nested loops over swept parameters the way process_countingstars_and_plot.m's is. It is
% four already-chosen operating points (KPAX, KinoPaxPlus, KinoPaxSTARTrue at ancestorPrune 1, and
% ONE CountingStars point), each run at all three deltas -- 12 series total, overlaid inside each
% figure. CountingStars runs bufferSlope 1.2, bufferFloor 0.4, explore_frac 0.15, cost_frac 0.75,
% WITH THE HOPELESS GUARD PERMANENTLY ON (v3.5, h_hopelessGuard_ in CountingStars.cuh) --
% countingstars_sweep.cu's own on/off sweep confirmed the guard helps at an earlier operating
% point, and slope/floor/ef/cf were re-tuned with it on (replacing the earlier two-point,
% unguarded bufferFloor 0.3/0.6 arm). KinoPaxSTARTrue (h_syclopCap_ at its 1.0 no-op default,
% h_ancestorPrune_ = 1) replaces KinoPaxSTARCleanCost as the non-CountingStars "STAR" reference
% this pass -- the naive OR-fusion of KPAX and KinoPaxPlus plus the guarded stale-best prune tells
% a "beats the naive fusion of its two parents" story, rather than CleanCost's "beats one already
% cost-tuned competitor." A second point at ancestorPrune 0 (the pure fusion, == stock
% KinoPaxSTARNoGoalBias exactly) ran alongside this one in an earlier pass to isolate what the
% guarded prune buys; it was dropped from this comparison (recoverable from git history as
% KinoPaxSTARTrue_cap100_anc0). Same three panels as
% process_countingstars_summary_plots.m (this script's direct ancestor -- loadRuns and every plot
% helper below are copies of its versions), plus a results table this one adds:
%
%   1. Best Cost vs Time        the fair cross-planner axis (an "iteration" is a different unit of
%                                work per planner; elapsed time is not).
%   2. Tree Growth vs Iteration how fast the tree actually fills, against nothing (there is no
%                                growth controller -- growth is an OUTPUT of the doors, not a
%                                target).
%   3. Tradeoff Scatter         mean time-to-first-solution vs mean FINAL cost, one point per
%                                (planner, delta). Lower-left wins both.
%   4. SUMMARY TABLE            one row per (planner, delta): time-to-first, cost-of-first,
%                                cost-of-last, success rate, final tree size %. Printed to the
%                                console AND written to a CSV alongside the figures.
%
% COLOR = PLANNER IDENTITY (4 fixed colors), WIDTH = DELTA (thin -> thick for large -> fine ->
% tiny). This replaces the swept-grid encoding (color=bufferFloor, style=bufferSlope,
% marker=(ef,cf)) that process_countingstars_summary_plots.m uses -- there is nothing left to sweep
% here, so color is free to carry the identity that actually varies across this comparison.
%
% USAGE: cd into ONE environment's data directory, then call the script BY NAME, not via run():
%   cd build/Data/Benchmarks/PaperBenchmarkV2/empty
%   addpath('<repo>/scripts')
%   process_paper_benchmark_v2_and_plot
% run('<abs path>/process_paper_benchmark_v2_and_plot.m') would cd to the scripts folder first, and
% dataDir below ('' = current folder) would then find nothing. Repeat once per environment,
% changing `environments` below to match the subfolder you cd'd into each time.

clear; clc; close all;

%% --- Configuration ---
dataDir = '';   % '' = current directory (run this from Data/Benchmarks/PaperBenchmarkV2/<env>)

% One environment per run -- must match the subfolder you cd'd into. Change this each time you
% move to a different Data/Benchmarks/PaperBenchmarkV2/<env> folder.
%
% ALL FOUR ENVIRONMENTS ARE VALID HERE (unlike process_paper_benchmark_and_plot.m, which excludes
% `empty` because paper_benchmark.cu hangs at tiny/empty) -- paper_benchmark_v2.cu runs on
% countingstars_sweep.cu's harness instead, which completes every delta including tiny cleanly.
environments = {'empty'};
envTitles    = {'Empty'};
% Other environments this suite produces (uncomment the one you cd'd into):
% environments = {'house'};          envTitles = {'House'};
% environments = {'narrowPassage'};  envTitles = {'Narrow Passage'};
% environments = {'zigzag'};         envTitles = {'Zigzag Corridor (tightened)'};

% Cost metric axis -- one build each, so one set of figures each. BOTH THIS PASS (see
% run_paper_benchmark_v2.sh's COST_LABELS; Dubins Airplane's own COST_MODE==1 effort branch in
% edgeCost() (helper.cuh) is what makes "effort" a genuinely different metric from "length" here).
metrics       = {'length', 'effort'};
metricTitles  = {'Workspace Path Length', 'Control Effort'};
metricYLabels = {'Path Cost (workspace path length)', 'Path Cost (control effort)'};
% metrics       = {'length'};
% metricTitles  = {'Workspace Path Length'};
% metricYLabels = {'Path Cost (workspace path length)'};

% Delta axis -- OVERLAID inside each figure, encoded as line WIDTH. The filename token is
% sprintf('%s_%s', delta, metric), e.g. 'fine_effort'. "fine" and "tiny" refine different axes
% (workspace vs. velocity) -- see run_paper_benchmark.sh -- and are NOT an identical-region-count
% pair (262,144 vs 592,704).
%
% TINY CHANGED FROM (W_R1=17, V_R1=5, 614,000 regions) THIS PASS -- that discretization crashed
% with a cudaErrorIllegalAddress in the `empty` environment (see run_paper_benchmark.sh's header
% for the debugging history). It is replaced with countingstars_sweep.cu's own tiny (W_R1=14,
% V_R1=6, 592,704 regions), confirmed clean across all five series in a dedicated bug-isolation
% harness.
deltas      = {'large', 'fine', 'tiny'};
deltaTitles = {'9k', '262k W-refined', '593k V-refined'};
deltaWidths = [1.0, 1.8, 2.6];

maxTreeSize = 3000000;   % MAX_TREE_SIZE in config.h -- denominator for the table's Final Tree (%)

% --- The four FIXED series (not swept). Label tokens must match examples/gpu/paper_benchmark.cu's
% trueLabel() / countingStarsLabel() exactly: round(100 x float) for cap/bs/bf, round(1000 x
% float) for ef/cf, plain int for hopelessGuard. ---
trueCap          = 100;   % syclopCap 1.0 (no cap)
trueAncestorPrune = 1;    % the guarded-prune point; anc0 (pure fusion) was dropped from this pass

% CountingStars' ONE operating point -- bufferSlope 1.2, bufferFloor 0.4, explore_frac 0.15,
% cost_frac 0.75, hopelessGuard PERMANENTLY ON (v3.5, see h_hopelessGuard_ in CountingStars.cuh).
% Replaces the earlier two-point (bufferFloor 0.3/0.6), unguarded arm now that the guard is
% confirmed to help and the ramp/budget have been re-tuned around it.
csSlope = 120; csFloor = 40; csExplore = 150; csCost = 750; csHopelessGuard = 1;

baseNames = { ...
    'KPAX', ...
    'KinoPaxPlus', ...
    sprintf('KinoPaxSTARTrue_cap%d_anc%d', trueCap, trueAncestorPrune), ...
    sprintf('CountingStars_bs%d_bf%d_ef%d_cf%d_hg%d', csSlope, csFloor, csExplore, csCost, csHopelessGuard) ...
};
baseDisplay = { ...
    'KPAX', ...
    'KinoPaxPlus', ...
    'KinoPaxSTARTrue (naive, stale-best prune)', ...
    'CountingStars (slope 1.2, floor 0.4, ef 0.15, cf 0.75, hopeless guard ON)' ...
};
% color = planner identity: KPAX near-black, KinoPaxPlus blue, KinoPaxSTARTrue crimson,
% CountingStars amber -- each "family" reads as its own hue.
baseColors = [ ...
    0.10 0.10 0.10;    % KPAX
    0.20 0.40 0.80;    % KinoPaxPlus
    0.45 0.05 0.10;    % KinoPaxSTARTrue (crimson)
    0.85 0.55 0.10 ];  % CountingStars (amber)
baseMarkers = {'s', 'd', 'h', 'o'};

% --- Build the series arrays: (planner, delta) pairs, planner-major so the legend and table group
% all three deltas together per planner. ---
plannerNames    = {};
plannerDisplay  = {};
plannerColors   = [];
plannerStyles   = {};
plannerMarkers  = {};
plannerWidths   = [];
plannerBaseline = [];   % logical: drawn as the larger scatter marker (all true -- every series here
                         % is a headline comparison point, not a swept grid)
plannerDeltaIdx = [];   % index into `deltas`
plannerBaseIdx  = [];   % index into `baseNames`/`baseDisplay` -- table's Planner column

for si = 1:numel(baseNames)
    for di = 1:numel(deltas)
        dWidth = deltaWidths(di);
        dTag   = deltaTitles{di};

        plannerNames{end + 1}   = baseNames{si};                              %#ok<SAGROW>
        plannerDisplay{end + 1} = sprintf('%s [%s]', baseDisplay{si}, dTag);  %#ok<SAGROW>
        plannerColors(end + 1, :) = baseColors(si, :);                        %#ok<SAGROW>
        plannerStyles{end + 1}    = '-';                                      %#ok<SAGROW>
        plannerMarkers{end + 1}   = baseMarkers{si};                          %#ok<SAGROW>
        plannerWidths(end + 1)    = dWidth;                                   %#ok<SAGROW>
        plannerBaseline(end + 1)  = true;                                     %#ok<SAGROW>
        plannerDeltaIdx(end + 1)  = di;                                       %#ok<SAGROW>
        plannerBaseIdx(end + 1)   = si;                                       %#ok<SAGROW>
    end
end

numRunsPer = 20 * ones(1, numel(plannerNames));   % max runs searched (missing files skipped); the
                                                    % harness writes 5, this just leaves headroom
                                                    % for a manual rerun without editing this file

MAX_FLOAT_THRESH = 1e30;   % best_cost sentinel (MAX_FLOAT / INFINITY) -> NaN
numTimeSamples   = 500;

nPlanner = numel(plannerNames);
deltaLabel = '3 deltas overlaid (width-coded)';

%% ======================================================================
figNum = 0;
for ei = 1:numel(environments)
    env      = environments{ei};
    envTitle = envTitles{ei};

    for mi = 1:numel(metrics)
        metric    = metrics{mi};
        costTitle = metricTitles{mi};
        costYLab  = metricYLabels{mi};
        fprintf('\n=== Environment: %s | Cost metric: %s ===\n', env, costTitle);

        % --- Load every (planner, delta) series for this cost metric ---
        R = cell(1, nPlanner);
        for pi = 1:nPlanner
            tok   = sprintf('%s_%s', deltas{plannerDeltaIdx(pi)}, metric);
            R{pi} = loadRuns(dataDir, env, plannerNames{pi}, tok, numRunsPer(pi));
            fprintf('  %-34s %-18s : %d runs\n', plannerNames{pi}, ...
                    ['[' deltas{plannerDeltaIdx(pi)} ']'], numel(R{pi}));
        end

        %% ---------- FIGURE 1: Best Cost vs Time (mean lines, no bands) ----------
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Cost vs Time (%s)', envTitle, costTitle), ...
               'Position', [40 40 1180 700]);
        hold on;
        tmax = globalMaxTime(R);
        if tmax > 0
            ct = linspace(0, tmax, numTimeSamples);
            for pi = 1:nPlanner
                plotMeanTime(R{pi}, 'best_cost', ct, plannerColors(pi, :), ...
                             plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
            end
        end
        xlabel('Elapsed Time (ms)'); ylabel(costYLab); grid on;
        clickableLegend();
        title(sprintf('Best Cost vs Time \x2014 %s, %s, %s', envTitle, deltaLabel, costTitle), ...
              'FontWeight', 'bold');

        %% ---------- FIGURE 2: Tree Growth vs Iteration ----------
        % THERE IS NO GROWTH CONTROLLER -- this is an OUTPUT of however many candidates the doors
        % admitted, not a target the planner tracks against a reference line.
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Tree Growth (%s)', envTitle, costTitle), ...
               'Position', [70 70 1000 640]);
        hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) getCol(t, 'tree_size'), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        xlabel('Iteration'); ylabel('tree\_size'); grid on;
        clickableLegend();
        title(sprintf('Tree Growth vs Iteration \x2014 %s, %s, %s', envTitle, deltaLabel, costTitle), ...
              'FontWeight', 'bold');

        %% ---------- Aggregate summary metrics per planner (FIGURE 3 + the table) ----------
        mFirstSolTime = NaN(1, nPlanner);
        mFirstSolCost = NaN(1, nPlanner);
        mFinalCost    = NaN(1, nPlanner);
        mSuccessPct   = NaN(1, nPlanner);
        mFinalTreePct = NaN(1, nPlanner);
        for pi = 1:nPlanner
            runs = R{pi};
            fstVals = []; fscVals = []; fcVals = []; treeVals = [];
            nSuccess = 0; nTotal = 0;
            for ri = 1:numel(runs)
                if isempty(runs{ri}), continue; end
                nTotal = nTotal + 1;
                ft = firstSolTime(runs{ri}, MAX_FLOAT_THRESH);
                if ft >= 0
                    fstVals(end + 1) = ft;   %#ok<SAGROW>
                    nSuccess = nSuccess + 1;
                end
                fsc = firstSolCost(runs{ri}, MAX_FLOAT_THRESH);
                if ~isnan(fsc), fscVals(end + 1) = fsc; end %#ok<SAGROW>
                fc = finalCost(runs{ri}, MAX_FLOAT_THRESH);
                if ~isnan(fc), fcVals(end + 1) = fc; end %#ok<SAGROW>
                treeVals(end + 1) = runs{ri}.tree_size(end); %#ok<SAGROW>
            end
            if ~isempty(fstVals), mFirstSolTime(pi) = mean(fstVals); end
            if ~isempty(fscVals), mFirstSolCost(pi) = mean(fscVals); end
            if ~isempty(fcVals),  mFinalCost(pi)    = mean(fcVals);  end
            if nTotal > 0,        mSuccessPct(pi)   = 100 * nSuccess / nTotal; end
            if ~isempty(treeVals), mFinalTreePct(pi) = 100 * mean(treeVals) / maxTreeSize; end
        end

        %% ---------- FIGURE 3: Tradeoff Scatter, Time to First Solution vs Final Cost ----------
        costLims = mFinalCost(isfinite(mFinalCost));
        if numel(costLims) >= 2 && max(costLims) > min(costLims)
            pad      = 0.05 * (max(costLims) - min(costLims));
            costYLim = [min(costLims) - pad, max(costLims) + pad];
        else
            costYLim = [];   % nothing solved, or a single value: let MATLAB autoscale
        end

        markerKey = sprintf(['lower-left is better (fast and cheap); width = delta (thin->thick = ' ...
                             'large->fine->tiny); \x25a1 KPAX, \x25c7 KinoPaxPlus, ' ...
                             '\x2605 KinoPaxSTARTrue, \x25cb CountingStars']);

        figNum = figNum + 1;
        figure('Name', sprintf('%s - Tradeoff Scatter (%s)', envTitle, costTitle), ...
               'Position', [130 140 1180 700]);
        tradeoffScatter(mFirstSolTime, mFinalCost, plannerMarkers, plannerColors, ...
                        plannerBaseline, plannerDisplay, costYLim);
        xlabel('Avg Time to First Solution (ms)'); ylabel(sprintf('Avg Final %s', costYLab));
        title(sprintf(['Tuning Tradeoff: Time to First Solution vs FINAL Cost \x2014 %s, %s\n%s'], ...
                       envTitle, costTitle, markerKey), 'FontWeight', 'bold');

        %% ---------- TABLE: one row per (planner, delta) ----------
        fprintf('\n--- Summary Table: %s | %s ---\n', envTitle, costTitle);
        fprintf('%-28s %-18s %16s %14s %14s %11s %13s\n', ...
                'Planner', 'Delta', 'TimeToFirst(ms)', 'CostOfFirst', 'CostOfLast', 'Success(%)', 'FinalTree(%)');
        tPlanner = cell(nPlanner, 1);
        tDelta   = cell(nPlanner, 1);
        for pi = 1:nPlanner
            tPlanner{pi} = baseDisplay{plannerBaseIdx(pi)};
            tDelta{pi}   = deltas{plannerDeltaIdx(pi)};
            fprintf('%-28s %-18s %16.2f %14.4f %14.4f %11.1f %13.2f\n', ...
                    tPlanner{pi}, tDelta{pi}, mFirstSolTime(pi), mFirstSolCost(pi), mFinalCost(pi), ...
                    mSuccessPct(pi), mFinalTreePct(pi));
        end

        T = table(tPlanner, tDelta, mFirstSolTime(:), mFirstSolCost(:), mFinalCost(:), ...
                  mSuccessPct(:), mFinalTreePct(:), 'VariableNames', ...
                  {'Planner', 'Delta', 'TimeToFirst_ms', 'CostOfFirst', 'CostOfLast', ...
                   'SuccessPct', 'FinalTreePct'});
        csvName = sprintf('paper_benchmark_v2_table_%s_%s.csv', sanitize_name(env), metric);
        writetable(T, fullfile(dataDir, csvName));
        fprintf('Table written to: %s\n', csvName);
    end
end

fprintf('\nGenerated %d figures.\n', figNum);

%% ======================================================================
%% Helper functions (local functions; copies of process_countingstars_summary_plots.m's)
%% ======================================================================
function runs = loadRuns(dataDir, env, planner, delta, numRuns)
    % Load one (planner, delta) series' per-run CSVs; missing files are skipped.
    runs = {};
    for ri = 0:(numRuns - 1)
        switch planner
            case 'KinoPaxPlus'
                fn = sprintf('%s_delta%s_run%d.csv', env, delta, ri);
            case 'KPAX'
                fn = sprintf('%s_KPAX_delta%s_run%d.csv', env, delta, ri);
            otherwise
                if startsWith(planner, 'CountingStars') || startsWith(planner, 'KinoPaxSTAR')
                    fn = sprintf('%s_%s_delta%s_run%d.csv', env, planner, delta, ri);
                else
                    error('unknown planner %s', planner);
                end
        end
        fp = fullfile(dataDir, fn);
        if isfile(fp)
            runs{end + 1} = readtable(fp); %#ok<AGROW>
        end
    end
end

function tmax = globalMaxTime(runsCell)
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

function plotMeanTime(runs, col, commonTime, color, style, width, name)
    % Mean of column 'col' vs a shared time grid (previous-sample hold, held forward past a run's
    % last sample so the curve's right edge is the mean of each run's FINAL value, not just the
    % longest-lasting run's).
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
            row = interp1(t, v, commonTime, 'previous', NaN);
            row(commonTime > t(end)) = v(end);
            A(ri, :) = row;
        end
    end
    A  = sanitize(A);
    mu = mean(A, 1, 'omitnan');
    valid = ~isnan(mu);
    if ~any(valid), return; end
    plot(commonTime(valid), mu(valid), style, 'Color', color, ...
         'LineWidth', width, 'DisplayName', name);
end

function plotMeanIter(runs, valueFcn, color, style, width, name)
    % Mean of a per-ITERATION quantity across a series' runs, drawn as ONE line. Runs are RAGGED
    % (the timeout ends them at different iterations); the tail is trimmed once fewer than half
    % the runs still contribute, so the right edge does not silently degenerate into one long run.
    if isempty(runs), return; end
    vals = {};
    for ri = 1:numel(runs)
        if isempty(runs{ri}), continue; end
        v = valueFcn(runs{ri});
        if isempty(v) || all(isnan(v)), continue; end
        vals{end + 1} = v(:); %#ok<AGROW>
    end
    if isempty(vals), return; end
    n = max(cellfun(@numel, vals));
    A = NaN(numel(vals), n);
    for ri = 1:numel(vals)
        A(ri, 1:numel(vals{ri})) = vals{ri};
    end
    cnt  = sum(~isnan(A), 1);
    mu   = mean(A, 1, 'omitnan');
    keep = cnt >= max(1, ceil(numel(vals) / 2));
    if ~any(keep), return; end
    plot(find(keep), mu(keep), style, 'Color', color, 'LineWidth', width, 'DisplayName', name);
end

function clickableLegend()
    lgd = legend('Location', 'eastoutside', 'FontSize', 6);
    lgd.ItemHitFcn = @toggleSeries;
end

function toggleSeries(~, evt)
    h = evt.Peer;
    if strcmp(h.Visible, 'on')
        h.Visible = 'off';
    else
        h.Visible = 'on';
    end
end

function v = getCol(tbl, col)
    if any(strcmp(col, tbl.Properties.VariableNames))
        v = tbl.(col);
    else
        v = [];
    end
end

function A = sanitize(A)
    A(A > 1e30) = NaN;
    A(A == -1)  = NaN;
end

function s = sanitize_name(s)
    % Filesystem-safe token for a CSV filename -- environment names here are already plain
    % alphanumeric, but this guards against a future environment name with a space or slash.
    s = regexprep(s, '[^a-zA-Z0-9_-]', '_');
end

function t = firstSolTime(tbl, thresh)
    % Elapsed time (ms) at which this run first reached a finite (< thresh) best_cost;
    % -1 if it never found a solution.
    solIdx = find(tbl.best_cost < thresh, 1, 'first');
    if isempty(solIdx), t = -1; else, t = tbl.elapsed_time_ms(solIdx); end
end

function c = firstSolCost(tbl, thresh)
    % The cost OF the first solution: best_cost at the first finite (< thresh) row. best_cost is
    % already the RUNNING best, so no extra bookkeeping is needed -- this is the same one-line
    % pattern as firstSolTime, over a different column. NaN if the run never found a solution.
    solIdx = find(tbl.best_cost < thresh, 1, 'first');
    if isempty(solIdx), c = NaN; else, c = tbl.best_cost(solIdx); end
end

function c = finalCost(tbl, thresh)
    costs = tbl.best_cost;
    costs(costs > thresh) = NaN;
    v = costs(~isnan(costs));
    if isempty(v), c = NaN; else, c = v(end); end
end

function tradeoffScatter(x, y, markers, colors, isBaseline, labels, yLim)
    hold on;
    for pi = 1:numel(x)
        if isnan(x(pi)) || isnan(y(pi)), continue; end
        if isBaseline(pi), msz = 11; else, msz = 8; end
        plot(x(pi), y(pi), markers{pi}, ...
             'MarkerFaceColor', colors(pi, :), ...
             'MarkerEdgeColor', 'k', 'LineWidth', 0.5, ...
             'MarkerSize', msz, 'DisplayName', labels{pi});
    end
    grid on;
    if ~isempty(yLim), ylim(yLim); end
    clickableLegend();
end
