%% Paper Benchmark Plots v2 - fixed 4-planner comparison, 3 panels + summary table per (env, model)
% Reads per-iteration CSVs produced by examples/gpu/paper_benchmark_v2.cu (run via
% scripts/run_paper_benchmark_v2.sh) -- countingstars_sweep.cu's proven-reliable harness, minimally
% adapted (all four environments, single fixed operating points) to reproduce
% paper_benchmark.cu's own headline comparison without paper_benchmark.cu's unresolved hang at
% `tiny`. run_paper_benchmark_v2.sh now builds and runs ALL THREE vehicle models in one sweep
% (Double Integrator, Dubins Airplane, Quad), each at exactly ONE discretization ("tiny" -- no more
% large/fine/tiny sweep per model) -- see that script's own header for the full derivation. This
% script still plots ONE MODEL AT A TIME (via the MODEL TOGGLE below), matching its original
% design: Figures 1-2 plot raw cost/tree-size values directly, which are only comparable WITHIN one
% vehicle model (Quad's workspace is [0,100]^3, the other two are [0,1]^3 -- overlaying models on
% these axes would mix physically different units, the same reason
% process_paper_benchmark_improvement_scatter.m's grand average omits raw cost across models).
% Otherwise identical in structure/presentation to process_paper_benchmark_and_plot.m.
%
% A FIXED COMPARISON, not a sweep -- there is no grid here, so the series list below is NOT built
% from nested loops over swept parameters the way process_countingstars_and_plot.m's is. It is four
% already-chosen operating points (KPAX, KinoPaxPlus, KinoPaxSTARTrue at ancestorPrune 1, and ONE
% CountingStars point), each at the ONE discretization the toggled model actually produced -- 4
% series total (used to be 12, one per (planner, delta), back when each model still ran a
% large/fine/tiny sweep). CountingStars runs bufferSlope 1.2, bufferFloor 0.4, explore_frac 0.15,
% cost_frac 0.75, WITH THE HOPELESS GUARD PERMANENTLY ON (v3.5, h_hopelessGuard_ in
% CountingStars.cuh) -- countingstars_sweep.cu's own on/off sweep confirmed the guard helps at an
% earlier operating point, and slope/floor/ef/cf were re-tuned with it on (replacing the earlier
% two-point, unguarded bufferFloor 0.3/0.6 arm). KinoPaxSTARTrue (h_syclopCap_ at its 1.0 no-op
% default, h_ancestorPrune_ = 1) replaces KinoPaxSTARCleanCost as the non-CountingStars "STAR"
% reference this pass -- the naive OR-fusion of KPAX and KinoPaxPlus plus the guarded stale-best
% prune tells a "beats the naive fusion of its two parents" story, rather than CleanCost's "beats
% one already cost-tuned competitor." A second point at ancestorPrune 0 (the pure fusion, == stock
% KinoPaxSTARNoGoalBias exactly) ran alongside this one in an earlier pass to isolate what the
% guarded prune buys; it was dropped from this comparison (recoverable from git history as
% KinoPaxSTARTrue_cap100_anc0). Same three panels as process_countingstars_summary_plots.m (this
% script's direct ancestor -- loadRuns and every plot helper below are copies of its versions),
% plus a results table this one adds:
%
%   1. Best Cost vs Time        the fair cross-planner axis (an "iteration" is a different unit of
%                                work per planner; elapsed time is not).
%   2. Tree Growth vs Iteration how fast the tree actually fills, against nothing (there is no
%                                growth controller -- growth is an OUTPUT of the doors, not a
%                                target).
%   3. Tradeoff Scatter         mean time-to-first-solution vs mean FINAL cost, one point per
%                                planner. Lower-left wins both.
%   4. SUMMARY TABLE            one row per planner: time-to-first, cost-of-first, cost-of-last,
%                                success rate, final tree size %. Printed to the console AND
%                                written to a CSV alongside the figures.
%
% COLOR = PLANNER IDENTITY (4 fixed colors: Kino-PAX, Kino-PAX+, SimpleCombo, KinoPax*), MARKER =
% PLANNER IDENTITY too (X, +, circle, star). There is no delta/model axis left to encode via marker
% size any more -- every planner runs at exactly the one discretization the toggled model produces,
% so all points share one plain size.
%
% ALL THREE LEGENDS RENDER BELOW THEIR PLOT ('southoutside', not 'eastoutside') so figures stay
% narrow enough for a paper column.
%
% MODEL TOGGLE (below, near `modelId`) selects both which vehicle model's region-count numbers
% `deltaTitle` shows AND which model's CSVs actually get loaded (via the "m<N>_" filename prefix
% run_paper_benchmark_v2.sh now writes into every output filename) -- this must match whatever
% produced the CSVs being loaded, or every series below will silently report "0 runs".
%
% USAGE: cd into ONE environment's data directory, then call the script BY NAME, not via run() --
% and NOT via the MATLAB Editor's Run/F5 button, which silently cd's to this file's OWN folder
% first, exactly like run() does, even if you already cd'd to the data folder in the Command Window:
%   cd build/Data/Benchmarks/PaperBenchmarkV2/empty
%   addpath('<repo>/scripts')
%   process_paper_benchmark_v2_and_plot
% Repeat once per environment, changing `environments` below to match the subfolder you cd'd into
% each time -- and once per model, changing the MODEL TOGGLE below to match.

clear; clc; close all;

fprintf('Running from: %s\n', pwd);

%% --- Configuration ---
dataDir = '';   % '' = current directory (run this from Data/Benchmarks/PaperBenchmarkV2/<env>)

% Figure size, in pixels -- one width/height pair per figure so each can be tuned independently
% (their default sizes already differ on purpose: Figure 3 is tallest, Figure 2 narrowest).
FIG1_WIDTH = 1180; FIG1_HEIGHT = 760;   % Cost vs Time
FIG2_WIDTH = 1000; FIG2_HEIGHT = 700;   % Tree Growth
FIG3_WIDTH = 1180; FIG3_HEIGHT = 820;   % Tradeoff Scatter

% One environment per run -- must match the subfolder you cd'd into. Change this each time you
% move to a different Data/Benchmarks/PaperBenchmarkV2/<env> folder.
%
% ALL FOUR ENVIRONMENTS ARE VALID HERE (unlike process_paper_benchmark_and_plot.m, which excludes
% `empty` because paper_benchmark.cu hangs at tiny/empty) -- paper_benchmark_v2.cu runs on
% countingstars_sweep.cu's harness instead, which completes every environment cleanly.
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

% --- MODEL TOGGLE --- must match whichever MODEL (paper_benchmark_v2.cu / write_config() in
% run_paper_benchmark_v2.sh) actually produced the CSVs being loaded below. `modelId` feeds the
% "m<N>_" filename prefix run_paper_benchmark_v2.sh's TAG now writes into every output filename (so
% all three models' CSVs can coexist in the same environment folder without colliding) -- uncomment
% the one block matching your data; comment the other two.

% MODEL 3 -- 12D Non-Linear Quad (W_DIM=3/C_DIM=3/V_DIM=3, native [0,100] workspace scale) -- CURRENT
modelId    = 3;
modelTitle = '12D Non-Linear Quad';
deltaTitle = 'Tiny (373k)';

% MODEL 2 -- 6D Dubins Airplane (C_DIM=2/V_DIM=1) -- uncomment if plotting Dubins-era CSVs:
% modelId    = 2;
% modelTitle = '6D Dubins Airplane';
% deltaTitle = 'Tiny (593k)';

% MODEL 1 -- 6D Double Integrator (C_DIM=0/V_DIM=3) -- uncomment if plotting Double-Integrator CSVs:
% modelId    = 1;
% modelTitle = '6D Double Integrator';
% deltaTitle = 'Tiny (593k)';

% Each model now runs at exactly ONE discretization (no more large/fine/tiny sweep -- see
% run_paper_benchmark_v2.sh's MODEL_W_R1S/MODEL_C_R1S/MODEL_V_R1S), always labeled "tiny" on disk;
% `deltaLabel` below is the bare on-disk token, `deltaTitle` above is just the display string.
deltaLabel = 'tiny';

maxTreeSize = 3000000;   % MAX_TREE_SIZE in config.h -- denominator for the table's Final Tree (%)

% --- The four FIXED series (not swept). Label tokens must match examples/gpu/paper_benchmark_v2.cu's
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

% One series per planner now -- no more (planner, delta) pairs, since each model runs exactly one
% discretization. plannerNames/plannerDisplay/plannerColors/plannerMarkers line up 1:1 with
% baseNames/baseDisplay/baseColors/baseMarkers; kept as separate names anyway so the rest of this
% file didn't need touching beyond dropping the now-gone delta axis.
plannerNames   = baseNames;
plannerDisplay = baseDisplay;
plannerColors  = baseColors;
plannerMarkers = baseMarkers;

numRunsPer = 35 * ones(1, numel(plannerNames));   % max runs searched (missing files skipped); the
                                                    % harness writes 30, this just leaves headroom
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

        % --- Load every planner's one series for this cost metric. Token carries the model, so
        % different models' CSVs in this same environment folder don't collide (see
        % run_paper_benchmark_v2.sh's TAG). ---
        R = cell(1, nPlanner);
        for pi = 1:nPlanner
            tok   = sprintf('m%d_%s_%s', modelId, deltaLabel, metric);
            R{pi} = loadRuns(dataDir, env, plannerNames{pi}, tok, numRunsPer(pi));
            fprintf('  %-34s : %d runs\n', plannerNames{pi}, numel(R{pi}));
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

        % Success rate is NOT shown on the legends/plots -- it's still computed above
        % (mSuccessPct) and lives in the printed table and CSV export below.
        legendLabels = plannerDisplay;

        %% ---------- FIGURE 1: Best Cost vs Time (mean lines, +/-1 std band across runs) ----------
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Cost vs Time (%s, %s)', envTitle, costTitle, modelTitle), ...
               'Position', [40 40 FIG1_WIDTH FIG1_HEIGHT]);
        hold on;
        tmax = globalMaxTime(R);
        if tmax > 0
            ct = linspace(0, tmax, numTimeSamples);
            for pi = 1:nPlanner
                plotMeanTime(R{pi}, 'best_cost', ct, plannerColors(pi, :), ...
                             '-', 2.5, legendLabels{pi});
            end
        end
        xlabel('Elapsed Time (ms)'); ylabel(costYLab); grid on;
        clickableLegend();
        title(sprintf('Cost vs Time \x2014 %s, %s, %s (%s)', envTitle, costTitle, modelTitle, deltaTitle), ...
              'FontWeight', 'bold');

        %% ---------- FIGURE 2: Tree Growth vs Iteration ----------
        % THERE IS NO GROWTH CONTROLLER -- this is an OUTPUT of however many candidates the doors
        % admitted, not a target the planner tracks against a reference line.
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Tree Growth (%s, %s)', envTitle, costTitle, modelTitle), ...
               'Position', [70 70 FIG2_WIDTH FIG2_HEIGHT]);
        hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) getCol(t, 'tree_size'), ...
                         plannerColors(pi, :), '-', 2.5, legendLabels{pi});
        end
        xlabel('Iteration'); ylabel('tree\_size'); grid on;
        clickableLegend();
        title(sprintf('Tree Growth \x2014 %s, %s, %s (%s)', envTitle, costTitle, modelTitle, deltaTitle), ...
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
               'Position', [130 140 FIG3_WIDTH FIG3_HEIGHT]);
        tradeoffScatter(mFirstSolTime, mFinalCost, plannerMarkers, plannerColors, legendLabels, costYLim);
        xlabel('Avg Time to First Solution (ms)'); ylabel(sprintf('Avg Final %s', costYLab));
        title(sprintf('Time to First Solution vs Final Cost \x2014 %s, %s, %s (%s)', ...
                      envTitle, costTitle, modelTitle, deltaTitle), 'FontWeight', 'bold');

        %% ---------- TABLE: one row per planner ----------
        fprintf('\n--- Summary Table: %s | %s | %s ---\n', envTitle, costTitle, modelTitle);
        fprintf('%-28s %16s %14s %14s %11s %13s\n', ...
                'Planner', 'TimeToFirst(ms)', 'CostOfFirst', 'CostOfLast', 'Success(%)', 'FinalTree(%)');
        tPlanner = cell(nPlanner, 1);
        for pi = 1:nPlanner
            tPlanner{pi} = plannerDisplay{pi};
            fprintf('%-28s %16.2f %14.4f %14.4f %11.1f %13.2f\n', ...
                    tPlanner{pi}, mFirstSolTime(pi), mFirstSolCost(pi), mFinalCost(pi), ...
                    mSuccessPct(pi), mFinalTreePct(pi));
        end

        T = table(tPlanner, mFirstSolTime(:), mFirstSolCost(:), mFinalCost(:), ...
                  mSuccessPct(:), mFinalTreePct(:), 'VariableNames', ...
                  {'Planner', 'TimeToFirst_ms', 'CostOfFirst', 'CostOfLast', ...
                   'SuccessPct', 'FinalTreePct'});
        % modelId is part of the filename -- all three models now write into the SAME environment
        % folder, so without it, plotting model 2 after model 1 would silently overwrite model 1's
        % table (this is the exact per-model collision bug fixed for the raw per-run CSVs in
        % run_paper_benchmark_v2.sh, applied here to this script's own summary-table output).
        csvName = sprintf('paper_benchmark_v2_table_%s_%s_m%d.csv', sanitize_name(env), metric, modelId);
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
    % longest-lasting run's). Also draws a shaded +/-1 std band across runs at each time sample,
    % UNDER the mean line and excluded from the legend (HandleVisibility off) -- this is what "add
    % error bars" resolves to for a continuous multi-run mean curve: discrete errorbar() whiskers
    % at every one of numTimeSamples points, x4 planners, would be unreadable clutter, so a
    % continuous band carries the same "how much do runs vary" information instead.
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
    A     = sanitize(A);
    mu    = mean(A, 1, 'omitnan');
    sigma = std(A, 0, 1, 'omitnan');
    sigma(isnan(sigma)) = 0;   % exactly one contributing run at that sample has zero spread, not NaN
    valid = ~isnan(mu);
    if ~any(valid), return; end
    tv = commonTime(valid); muv = mu(valid); sv = sigma(valid);
    fill([tv, fliplr(tv)], [muv + sv, fliplr(muv - sv)], color, ...
         'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(tv, muv, style, 'Color', color, 'LineWidth', width, 'DisplayName', name);
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
    % numCols: optional column count to wrap a big legend into multiple rows below the plot
    % instead of one very wide row. Omit/pass [] for a plain single-row legend -- all three
    % figures here have exactly 4 entries now, so this is rarely needed, but kept for headroom.
    if nargin < 1
        numCols = [];
    end
    lgd = legend('Location', 'southoutside', 'FontSize', 6);

    % Orientation='horizontal' is a long-standing legend property. NumColumns -- the property that
    % actually WRAPS a horizontal legend into multiple rows once numCols is given -- was only
    % added in MATLAB R2018a, so it gets the same defensive try/catch as ItemTokenSize below.
    try
        if ~isempty(numCols)
            lgd.NumColumns = numCols;
        end
        lgd.Orientation = 'horizontal';
    catch
        lgd.Orientation = 'vertical';
    end

    % ItemTokenSize widens the little icon swatch each legend row draws its line/marker preview in
    % -- not needed for delta/model-size encoding any more (each model now runs exactly one
    % discretization), kept for general marker legibility. NOT SUPPORTED on MATLAB releases before
    % it was added (errors as "Unknown property" if passed to the legend(...) constructor itself,
    % which is why this is a try/catch property SET afterward, not a constructor argument --
    % silently no-ops on any release that lacks it).
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

function tradeoffScatter(x, y, markers, colors, labels, yLim)
    % One point per planner -- no more delta/model size-coding (each model now runs exactly one
    % discretization; this script plots one model at a time via the MODEL TOGGLE), so every point
    % shares one plain size.
    MARKER_SIZE = 12;
    hold on;
    for pi = 1:numel(x)
        if isnan(x(pi)) || isnan(y(pi)), continue; end
        m = markers{pi};
        if m == 'x' || m == '+'
            % Stroke-only markers ('x','+') have no fillable interior -- MATLAB ignores
            % MarkerFaceColor for them entirely, so their visible color comes ONLY from
            % MarkerEdgeColor. A much thicker stroke is also what makes a '+' actually read as a
            % bold cross instead of a thin plus sign.
            plot(x(pi), y(pi), m, 'MarkerEdgeColor', colors(pi, :), 'LineWidth', 3.5, ...
                 'MarkerSize', MARKER_SIZE, 'DisplayName', labels{pi});
        else
            plot(x(pi), y(pi), m, ...
                 'MarkerFaceColor', colors(pi, :), ...
                 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, ...
                 'MarkerSize', MARKER_SIZE, 'DisplayName', labels{pi});
        end
    end
    grid on;
    if ~isempty(yLim), ylim(yLim); end
    clickableLegend();
end
