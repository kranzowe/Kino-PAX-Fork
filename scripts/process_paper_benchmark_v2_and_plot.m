%% Paper Benchmark Plots v2 - fixed 4-planner comparison, 3 panels + summary table per (env, metric)
% Reads per-iteration CSVs produced by examples/gpu/paper_benchmark_v2.cu (run via
% scripts/run_paper_benchmark_v2.sh) -- countingstars_sweep.cu's proven-reliable harness, minimally
% adapted (all four environments, single fixed operating points) to reproduce
% paper_benchmark.cu's own headline comparison without paper_benchmark.cu's unresolved hang at
% `tiny`. MODEL has moved twice since: first to 2 (Dubins Airplane, matching paper_benchmark.cu's
% original target but with a corrected C_DIM=2/V_DIM=1 dimension breakdown instead of
% paper_benchmark.cu's own C_DIM=0/V_DIM=3), and now to 3 (12D Non-Linear Quad, W_DIM=3/C_DIM=3/
% V_DIM=3, native [0,100] workspace scale) -- so results here are NOT directly comparable to
% paper_benchmark.cu's own numbers. See paper_benchmark_v2.cu / run_paper_benchmark_v2.sh for the
% full derivation of both switches. Otherwise identical in structure/presentation to
% process_paper_benchmark_and_plot.m.
%
% A FIXED COMPARISON, not a sweep -- there is no grid here, so the series list below is NOT built
% from nested loops over swept parameters the way process_countingstars_and_plot.m's is. It is
% four already-chosen operating points (KPAX, KinoPaxPlus, KinoPaxSTARTrue at ancestorPrune 1, and
% ONE CountingStars point), each run at all three deltas -- 12 series total. Figures 1-2 (cost vs
% time, tree growth) only plot the "fine" delta, to keep those two clean; Figure 3 (the tradeoff
% scatter) is the one place all three deltas are overlaid, coded by marker size. CountingStars
% runs bufferSlope 1.2, bufferFloor 0.4, explore_frac 0.15, cost_frac 0.75,
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
% COLOR = PLANNER IDENTITY (4 fixed colors: Kino-PAX, Kino-PAX+, SimpleCombo, KinoPax*), MARKER =
% PLANNER IDENTITY too (X, +, circle, star). DELTA only appears as an encoded dimension in Figure
% 3's scatter, as marker SIZE -- Figures 1-2 are solid lines at a single delta ("fine") instead of
% overlaying all three, since that reads far more cleanly. This replaces the swept-grid encoding
% (color=bufferFloor, style=bufferSlope, marker=(ef,cf)) that process_countingstars_summary_plots.m
% uses -- there is nothing left to sweep here, so color is free to carry planner identity.
%
% ALL THREE LEGENDS RENDER BELOW THEIR PLOT ('southoutside', not 'eastoutside') so figures stay
% narrow enough for a paper column. Figure 3's marker SIZE still encodes delta -- see
% clickableLegend()'s ItemTokenSize comment for the legend-icon-scaling caveat and which MATLAB
% release actually needs to fix it.
%
% MODEL TOGGLE (below, near `deltas`) selects which vehicle model's region-count numbers
% `deltaTitles` shows -- the delta TOKENS are the same text for every model, but the real region
% counts behind 'large'/'fine'/'tiny' differ per model, so this must match whatever produced the
% CSVs being loaded.
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
% run_paper_benchmark_v2.sh's COST_LABELS). CAUTION under MODEL 3 (Quad): edgeCost() (helper.cuh)
% has a real COST_MODE==1 branch for MODEL 1 and MODEL 2 (Dubins), but none for MODEL 3 -- so for
% Quad, "effort" silently falls back to the same workspace-distance formula as "length", and the
% two metrics' figures/table rows will be identical. Not wrong, just redundant; drop 'effort' below
% if that's not wanted while running Quad.
metrics       = {'length', 'effort'};
metricTitles  = {'Workspace Path Length', 'Control Effort'};
metricYLabels = {'Path Cost (workspace path length)', 'Path Cost (control effort)'};
% metrics       = {'length'};
% metricTitles  = {'Workspace Path Length'};
% metricYLabels = {'Path Cost (workspace path length)'};

% Delta axis. The filename token is sprintf('%s_%s', delta, metric), e.g. 'fine_effort'. Delta
% TOKENS ('large'/'fine'/'tiny') are identical text across every model that has ever produced
% these CSVs -- but the REGION COUNTS behind each token are NOT, since each model discretizes a
% different W/C/V shape (see the MODEL TOGGLE below, which is what actually needs to change per
% model). Figures 1-2 only ever load/plot the "fine" entry (fineIdx below); Figure 3's scatter is
% the only place all three are overlaid, encoded as marker SIZE.
deltas  = {'large', 'fine', 'tiny'};
fineIdx = find(strcmp(deltas, 'fine'));   % the one delta Figures 1-2 actually plot

% --- MODEL TOGGLE --- must match whichever MODEL (paper_benchmark_v2.cu / write_config() in
% run_paper_benchmark_v2.sh) actually produced the CSVs being loaded below. Uncomment the one
% block matching your data; comment the other two.

% MODEL 3 -- 12D Non-Linear Quad (W_DIM=3/C_DIM=3/V_DIM=3, native [0,100] workspace scale) -- CURRENT
modelTitle  = '12D Non-Linear Quad';
deltaTitles = {'Large (8k)', 'Fine (216k)', 'Tiny (373k)'};

% MODEL 2 -- 6D Dubins Airplane (C_DIM=2/V_DIM=1) -- uncomment if plotting Dubins-era CSVs:
% modelTitle  = '6D Dubins Airplane';
% deltaTitles = {'Large (9k)', 'Fine (262k)', 'Tiny (593k)'};

% MODEL 1 -- 6D Double Integrator (C_DIM=0/V_DIM=3) -- uncomment if plotting Double-Integrator CSVs:
% modelTitle  = '6D Double Integrator';
% deltaTitles = {'Large (9k)', 'Fine (262k)', 'Tiny (593k)'};

% Marker SIZE is the only place delta is visually encoded now (Figure 3's scatter, which overlays
% all three) -- deliberately dramatic so it reads at a glance.
deltaMarkerSizes = [7, 11, 16];

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
% Display names, no algorithm-parameter parentheticals -- the params (syclopCap/ancestorPrune,
% bufferSlope/bufferFloor/ef/cf/hopelessGuard) still fully identify each series via `baseNames`
% above (which is what loadRuns() actually matches against); they just don't clutter the legend.
baseDisplay = { ...
    'Kino-PAX', ...
    'Kino-PAX+', ...
    'SimpleCombo', ...
    'KinoPax*' ...
};
% color = planner identity: Kino-PAX near-black, Kino-PAX+ vivid blue, SimpleCombo vivid red,
% KinoPax* vivid yellow -- each "family" reads as its own hue, saturated enough to pop.
baseColors = [ ...
    0.10 0.10 0.10;    % Kino-PAX
    0.00 0.40 0.95;    % Kino-PAX+ (vivid blue)
    0.85 0.05 0.05;    % SimpleCombo (vivid red)
    0.95 0.70 0.00 ];  % KinoPax* (vivid yellow/gold)
% marker = planner identity too: X, +, circle, star -- matches the display names above (Kino-PAX
% is the "X" mark, Kino-PAX+ the "+" mark, SimpleCombo a plain circle, KinoPax* an actual star).
baseMarkers = {'x', '+', 'o', 'p'};

% --- Build the series arrays: (planner, delta) pairs, planner-major so the legend and table group
% all three deltas together per planner. ---
plannerNames    = {};
plannerDisplay  = {};
plannerColors   = [];
plannerMarkers  = {};
plannerSizes    = [];   % scatter marker size, delta-coded (deltaMarkerSizes) -- every series here
                         % is a headline comparison point, not a swept grid, so size carries delta
                         % rather than a baseline/swept-point distinction
plannerDeltaIdx = [];   % index into `deltas`
plannerBaseIdx  = [];   % index into `baseNames`/`baseDisplay` -- table's Planner column

for si = 1:numel(baseNames)
    for di = 1:numel(deltas)
        dTag = deltaTitles{di};

        plannerNames{end + 1}   = baseNames{si};                              %#ok<SAGROW>
        plannerDisplay{end + 1} = sprintf('%s [%s]', baseDisplay{si}, dTag);  %#ok<SAGROW>
        plannerColors(end + 1, :) = baseColors(si, :);                        %#ok<SAGROW>
        plannerMarkers{end + 1}   = baseMarkers{si};                          %#ok<SAGROW>
        plannerSizes(end + 1)     = deltaMarkerSizes(di);                    %#ok<SAGROW>
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

%% ======================================================================
figNum = 0;
for ei = 1:numel(environments)
    env      = environments{ei};
    envTitle = envTitles{ei};

    for mi = 1:numel(metrics)
        metric    = metrics{mi};
        costTitle = metricTitles{mi};
        costYLab  = metricYLabels{mi};
        fprintf('\n=== Environment: %s | Cost metric: %s | Model: %s ===\n', env, costTitle, modelTitle);

        % --- Load every (planner, delta) series for this cost metric ---
        R = cell(1, nPlanner);
        for pi = 1:nPlanner
            tok   = sprintf('%s_%s', deltas{plannerDeltaIdx(pi)}, metric);
            R{pi} = loadRuns(dataDir, env, plannerNames{pi}, tok, numRunsPer(pi));
            fprintf('  %-34s %-18s : %d runs\n', plannerNames{pi}, ...
                    ['[' deltas{plannerDeltaIdx(pi)} ']'], numel(R{pi}));
        end

        %% ---------- Aggregate summary metrics per planner (drives the legends, FIGURE 3, table) ----------
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

        % `legendLabels` keeps the [delta] tag (Figure 3, which overlays all three deltas);
        % `legendLabelsFine` drops it (Figures 1-2, which only ever show the one "fine" series per
        % planner, so repeating the same tag on every line would just be noise). Success rate is
        % NOT shown here -- it's still computed above (mSuccessPct) and lives in the printed table
        % and CSV export below, just not on the plots themselves.
        legendLabels     = plannerDisplay;
        legendLabelsFine = cell(1, nPlanner);
        for pi = 1:nPlanner
            legendLabelsFine{pi} = baseDisplay{plannerBaseIdx(pi)};
        end

        %% ---------- FIGURE 1: Best Cost vs Time (mean lines, no bands, "fine" delta only) ----------
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Cost vs Time (%s, %s)', envTitle, costTitle, modelTitle), ...
               'Position', [40 40 1180 760]);
        hold on;
        tmax = globalMaxTime(R(plannerDeltaIdx == fineIdx));
        if tmax > 0
            ct = linspace(0, tmax, numTimeSamples);
            for pi = 1:nPlanner
                if plannerDeltaIdx(pi) ~= fineIdx, continue; end
                plotMeanTime(R{pi}, 'best_cost', ct, plannerColors(pi, :), ...
                             '-', 2.5, legendLabelsFine{pi});
            end
        end
        xlabel('Elapsed Time (ms)'); ylabel(costYLab); grid on;
        clickableLegend();
        title(sprintf('Cost vs Time \x2014 %s, %s, %s', envTitle, costTitle, deltaTitles{fineIdx}), ...
              'FontWeight', 'bold');

        %% ---------- FIGURE 2: Tree Growth vs Iteration ("fine" delta only) ----------
        % THERE IS NO GROWTH CONTROLLER -- this is an OUTPUT of however many candidates the doors
        % admitted, not a target the planner tracks against a reference line.
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Tree Growth (%s, %s)', envTitle, costTitle, modelTitle), ...
               'Position', [70 70 1000 700]);
        hold on;
        for pi = 1:nPlanner
            if plannerDeltaIdx(pi) ~= fineIdx, continue; end
            plotMeanIter(R{pi}, @(t) getCol(t, 'tree_size'), ...
                         plannerColors(pi, :), '-', 2.5, legendLabelsFine{pi});
        end
        xlabel('Iteration'); ylabel('tree\_size'); grid on;
        clickableLegend();
        title(sprintf('Tree Growth \x2014 %s, %s, %s', envTitle, costTitle, deltaTitles{fineIdx}), ...
              'FontWeight', 'bold');

        %% ---------- FIGURE 3: Tradeoff Scatter, Time to First Solution vs Final Cost ----------
        costLims = mFinalCost(isfinite(mFinalCost));
        if numel(costLims) >= 2 && max(costLims) > min(costLims)
            pad      = 0.05 * (max(costLims) - min(costLims));
            costYLim = [min(costLims) - pad, max(costLims) + pad];
        else
            costYLim = [];   % nothing solved, or a single value: let MATLAB autoscale
        end

        figNum = figNum + 1;
        figure('Name', sprintf('%s - Tradeoff Scatter (%s, %s)', envTitle, costTitle, modelTitle), ...
               'Position', [130 140 1180 820]);
        tradeoffScatter(mFirstSolTime, mFinalCost, plannerMarkers, plannerColors, ...
                        plannerSizes, legendLabels, costYLim);
        xlabel('Avg Time to First Solution (ms)'); ylabel(sprintf('Avg Final %s', costYLab));
        title(sprintf('Time to First Solution vs Final Cost \x2014 %s, %s, %s', envTitle, costTitle, modelTitle), ...
              'FontWeight', 'bold');

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

function clickableLegend(numCols)
    % numCols: optional column count to wrap a big legend (Figure 3's 12 entries) into multiple
    % rows below the plot instead of one very wide row. Omit/pass [] for a plain single-row
    % legend (Figures 1-2's 4 entries).
    if nargin < 1
        numCols = [];
    end
    lgd = legend('Location', 'southoutside', 'FontSize', 6);

    % Orientation='horizontal' is a long-standing legend property. NumColumns -- the property that
    % actually WRAPS a horizontal legend into multiple rows once numCols is given -- was only
    % added in MATLAB R2018a, so it gets the same defensive try/catch as ItemTokenSize below.
    % Both are set inside the SAME try: if NumColumns throws on an older release, fall back to a
    % VERTICAL (single column, just tall) legend instead of ending up horizontal-but-unwrapped,
    % which could render one legend row wider than the whole figure with 12 entries.
    try
        if ~isempty(numCols)
            lgd.NumColumns = numCols;
        end
        lgd.Orientation = 'horizontal';
    catch
        lgd.Orientation = 'vertical';
    end

    % ItemTokenSize widens the little icon swatch each legend row draws its line/marker preview
    % in -- MATLAB's default ([30 18]-ish) clips a big marker down to look the same size as a
    % small one, which is exactly the "legend symbols don't scale like on the plot" problem.
    % NOT SUPPORTED on MATLAB releases before it was added (errors as "Unknown property" if passed
    % to the legend(...) constructor itself, which is why this is a try/catch property SET
    % afterward, not a constructor argument -- silently no-ops on any release that lacks it). This
    % is the only mechanism left for delta-size legend scaling (the earlier hand-built size-key
    % inset was removed as redundant once this property is available).
    try
        lgd.ItemTokenSize = [45, 24];
    catch
    end
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

function tradeoffScatter(x, y, markers, colors, sizes, labels, yLim)
    % sizes is delta-coded (deltaMarkerSizes), not a baseline/swept-point distinction -- every
    % series in this comparison is a headline point, so size is free to carry delta instead.
    % Delta-size legend scaling relies on clickableLegend()'s ItemTokenSize (see its comment) --
    % there is no hand-built fallback key here any more.
    hold on;
    for pi = 1:numel(x)
        if isnan(x(pi)) || isnan(y(pi)), continue; end
        m = markers{pi};
        if m == 'x' || m == '+'
            % Stroke-only markers ('x','+') have no fillable interior -- MATLAB ignores
            % MarkerFaceColor for them entirely, so their visible color comes ONLY from
            % MarkerEdgeColor (previously hardcoded 'k', i.e. every one of these was rendering
            % black regardless of its assigned planner color). A much thicker stroke is also what
            % makes a '+' actually read as a bold cross instead of a thin plus sign.
            plot(x(pi), y(pi), m, 'MarkerEdgeColor', colors(pi, :), 'LineWidth', 3.5, ...
                 'MarkerSize', sizes(pi), 'DisplayName', labels{pi});
        else
            plot(x(pi), y(pi), m, ...
                 'MarkerFaceColor', colors(pi, :), ...
                 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, ...
                 'MarkerSize', sizes(pi), 'DisplayName', labels{pi});
        end
    end
    grid on;
    if ~isempty(yLim), ylim(yLim); end

    n = numel(labels);
    if n > 6
        numCols = ceil(n / 3);   % wrap a big legend (Figure 3's 12 entries) into ~3 rows
    else
        numCols = [];
    end
    clickableLegend(numCols);
end
