%% Cost Tuning Sweep Visualization — w x k x cap grid, both cost metrics
% Reads per-iteration CSVs produced by kinopaxstar_cost_tuning_sweep.cu
% (run via run_cost_tuning_sweep.sh).
%
% THREE DELTAS ARE OVERLAID in every figure. Series are (planner, delta) pairs:
%
%   delta 'large' (27k regions, full cap sweep)                                = 14
%       CleanCost w {0.9} x k {8} x cap {0.01, 0.03, 0.05, 0.1}                     4
%       TrueStar  cap {0.01, 0.03, 0.05, 0.1}                                       4
%       KPAXCap   cap {0.01, 0.03, 0.05, 0.1}                                       4
%       KPAX, KinoPaxPlus                                                           2
%   delta 'fine' (216k, workspace-refined; --single-cap, derived cap only)     =  5
%   delta 'fine_control' (216k, velocity-refined; --single-cap)                =  5
%                                                                              -------
%                                                                                  24
%
% w and k are both PINNED (0.9, 8), so this is a pure cap sweep. k was measured to be inert and the
% formula says why: with w = 0.9 and P_floor = EPSILON = 0.01 the cost term outranks the floor only
% while x < ln(10)/k for x = (cost-m)/(mean-m), and the region minimum is exempt anyway via the
% cost <= m early return.
%
% cap = 0.03 is the DERIVED operating point (~1/h_activeBlockSize_ = 1/32): after the acceptance
% fold each frontier node offers repeat*blockSize candidates to one rule, so holding the per-node
% branching factor near 1 gives cap ~ 1/blockSize. The finer deltas run only that cap, so all three
% deltas overlay at a matched point.
%
% 'fine' and 'fine_control' are a CONTROLLED PAIR: identical 216,000 region count, refined in
% workspace (W_R1 10->20) vs velocity (V_R1 3->6). C_R1 is inert under C_DIM 0 and stays at 1.
%
% KPAXCap is stock KPAX with the SAME cap multiplier on the Syclop score, applied at both
% acceptance points. It is the control arm for the cap: CleanCost at w = 1 applies the cap AND
% decides after graph_.updateVertices(), so KPAX / KPAXCap / CleanCost-at-w=1 separates the cap's
% effect from the kernel boundary's.
%
% CleanCost makes exactly ONE acceptance decision, in the accept kernel, where the region cost
% statistics have converged and vertexScores already include the current iteration's samples:
%     P = cap * min(1, w*(vertexScore + fAccept) + (1-w)*costProbExp(k) + P_floor)
% with region-best and fresh-R2-sub-region candidates exempt. Its predecessor
% KinoPaxSTARWeightedCost also ran a propagate-time filter capped at 0.1 that sat silently upstream
% of w; cap is the explicit replacement, applied at both the accept kernel and Part-B reactivation.
% w = 1 reproduces KPAX's acceptance, w = 0 is pure cost-greedy. P_cost is
% exp(-k*(cost-m)/(mean-m)): exactly 1 at the region min AND with a real gradient across the whole
% range, unlike min(1,(mean/cost)^k), which is pinned at 1 for every cost at or below the mean.
%
% TrueStar keeps the plain KPAX Syclop roll with the region score scaled by cap (fAccept unscaled),
% plus the guarded stale-best cost prune.
%
% TWO DISCRETIZATIONS. KinoPaxPlus is measured twice: at the sweep's own delta ("large", 27k R1
% regions) and at a finer one ("fine", 216k) built as a separate binary, since NUM_R1_REGIONS is
% compile-time. Only KinoPaxPlus runs at the fine delta.
%
% ONE ENVIRONMENT PER RUN. The benchmark writes each environment to its own subfolder
% (Data/Benchmarks/KinoPaxStarCostTuning/<env>/), so cd into the one you want and set envName
% below to match.
%
% COST METRIC: swept by rebuilding, since COST_MODE is a compile-time #if inside
% edgeCost (include/helper/helper.cuh). The metric therefore rides in the delta
% token of every filename: large_effort (control effort) vs large_length (workspace
% path length). The two metrics have DIFFERENT UNITS, so every cost-dependent
% figure below is drawn once per metric and they are never overlaid.
%
% Per-run CSVs (in dataDir):
%   KinoPaxPlus:         {env}_delta{large_METRIC}_run{n}.csv
%   KinoPaxPlus (fine):  {env}_delta{fine_METRIC}_run{n}.csv
%   KPAX:                {env}_KPAX_delta{large_METRIC}_run{n}.csv
%   STAR variants:       {env}_{planner label}_delta{large_METRIC}_run{n}.csv
%                        e.g. KinoPaxSTARCleanCost_w90_k400_cap5, KinoPaxSTARTrue_cap5
%
% Per-iteration columns: iteration, frontier_size, tree_size, elapsed_time_ms, best_cost
%
% ENCODING: k is pinned, which frees the line-style channel for DELTA -- '-' large, '--' fine,
% ':' fine_control. Colour keeps its planner+cap meaning: steel-blue ramp = CleanCost (light->dark
% as cap goes 0.01->0.1), amber = TrueStar, grey-green = KPAXCap, near-black = KPAX, blue =
% KinoPaxPlus. So a COLOUR is one planner at one cap across all three deltas, and a STYLE is one
% delta across all planners -- which makes delta-vs-delta the easy read. Every legend here is
% CLICKABLE — click an entry to hide/show that series.
%
% FAIR-COMPARISON NOTE: an "iteration" is a different unit of work per planner, so
% cost-vs-TIME is the fair cross-planner axis. Error bands and error bars are
% deliberately omitted throughout; the scatter shows run means only.
%
% USAGE: cd into the data directory, then call the script BY NAME, not via run():
%   cd build/Data/Benchmarks/KinoPaxStarCostTuning/zigzag     % or .../house
%   addpath('<repo>/scripts')
%   process_cost_tuning_and_plot
% run('<abs path>/process_cost_tuning_and_plot.m') would cd to the scripts folder
% first, and dataDir below ('' = current folder) would then find nothing.

clear; clc; close all;

%% --- Configuration ---
dataDir = '';   % '' = current directory (run this from Data/Benchmarks/KinoPaxStarCostTuning)

% One environment per run — must match the subfolder you cd'd into.
%   'zigzag' -> 'Zigzag Corridor',  'house' -> 'House'
environments = {'zigzag'};
envTitles    = {'Zigzag Corridor'};

% Cost metric axis — one build each, so one set of figures each.
metrics      = {'effort', 'length'};
metricTitles = {'Control Effort', 'Workspace Path Length'};
metricYLabels = {'Path Cost (control effort)', 'Path Cost (workspace path length)'};

% Delta axis — OVERLAID inside each figure, encoded as line style. The filename token is
% sprintf('%s_%s', delta, metric), e.g. 'fine_control_length'.
deltas      = {'large', 'fine', 'fine_control'};
deltaTitles = {'27k', '216k W-refined', '216k V-refined'};
deltaStyles = {'-', '--', ':'};
% Which caps exist at each delta: the coarse delta sweeps the axis, the finer ones ran --single-cap
% so only the derived point exists. Must match DELTA_EXTRA_ARGS in run_cost_tuning_sweep.sh.
capDerived  = 3;                       % label token for cap = 0.03 (~1/blockSize)
deltaCaps   = {[1 3 5 10], capDerived, capDerived};

deltaLabel = '3 deltas overlaid';

% CleanCost grid — must match WEIGHTS / WEIGHTED_EXPS / CAPS in
% kinopaxstar_cost_tuning_sweep.cu. Values are the integer label tokens (100 x the float),
% exactly as they appear in the filenames.
% CleanCost grid — must match WEIGHTS / WEIGHTED_EXPS / CAPS in the benchmark. Values are the
% integer label tokens (100 x the float), exactly as they appear in the filenames.
weights = [90];
wExps   = [800];
caps    = [1 3 5 10];

% TrueStar and KPAXCap cap sweeps — must match TRUE_CAPS / KPAXCAP_CAPS in the benchmark.
% Deliberately the same cap values as the CleanCost grid, so a cap is comparable across planners.
trueCaps    = [1 3 5 10];
kpaxCapCaps = [1 3 5 10];

% Colour = planner + cap (light->dark as cap grows). Line style = delta. Width = w (constant).
capRamp    = [0.70 0.82 0.93;    % cap 0.01 - steel blue (lightest)
              0.45 0.64 0.84;    % cap 0.03  <- the derived operating point
              0.21 0.42 0.66;    % cap 0.05
              0.03 0.15 0.31];   % cap 0.10 - steel blue (darkest)
wWidths    = [1.4];              % one entry per w; widens the channel again if w is swept

% TrueStar gets its own warm ramp over the same cap values.
amberRamp = [0.99 0.85 0.62;     % cap 0.01 (lightest)
             0.97 0.68 0.33;
             0.88 0.47 0.10;
             0.60 0.28 0.02];    % cap 0.10 (darkest)

% KPAXCap: grey-green, distinct from both the steel ramp and near-black KPAX it is compared to.
mossRamp  = [0.72 0.83 0.68;     % cap 0.01 (lightest)
             0.55 0.70 0.50;
             0.36 0.56 0.36;
             0.18 0.36 0.20];    % cap 0.10 (darkest)

% --- Build the series arrays: (planner, delta) pairs, 24 in total ---
% plannerDeltaIdx carries each series' delta so loadRuns can build its own filename token; the
% style channel is delta, so every series of one delta shares a line style.
plannerNames    = {};
plannerDisplay  = {};
plannerColors   = [];
plannerStyles   = {};
plannerMarkers  = {};
plannerWidths   = [];
plannerBaseline = [];   % logical: drawn as a thick reference anchor / large scatter marker
plannerDeltaIdx = [];   % index into `deltas`

for di = 1:numel(deltas)
    dStyle = deltaStyles{di};
    dTag   = deltaTitles{di};
    dCaps  = deltaCaps{di};    % caps that actually exist at this delta

    % --- CleanCost: w x k x cap, restricted to this delta's caps ---
    for wi = 1:numel(weights)
        for ei = 1:numel(wExps)
            for ci = 1:numel(caps)
                if ~ismember(caps(ci), dCaps), continue; end
                % Mirror the benchmark's skip: at w = 1 the cost term vanishes, so only k = 1 runs.
                if weights(wi) == 100 && wExps(ei) ~= 100, continue; end
                plannerNames{end + 1}   = sprintf('KinoPaxSTARCleanCost_w%d_k%d_cap%d', ...
                                                  weights(wi), wExps(ei), caps(ci)); %#ok<SAGROW>
                plannerDisplay{end + 1} = sprintf('C cap%g [%s]', caps(ci) / 100, dTag); %#ok<SAGROW>
                plannerColors(end + 1, :) = capRamp(ci, :);   %#ok<SAGROW>
                plannerStyles{end + 1}    = dStyle;           %#ok<SAGROW>
                plannerMarkers{end + 1}   = 'o';              %#ok<SAGROW>
                plannerWidths(end + 1)    = wWidths(wi);      %#ok<SAGROW>
                plannerBaseline(end + 1)  = false;            %#ok<SAGROW>
                plannerDeltaIdx(end + 1)  = di;               %#ok<SAGROW>
            end
        end
    end

    % --- TrueStar ---
    for ci = 1:numel(trueCaps)
        if ~ismember(trueCaps(ci), dCaps), continue; end
        plannerNames{end + 1}   = sprintf('KinoPaxSTARTrue_cap%d', trueCaps(ci)); %#ok<SAGROW>
        plannerDisplay{end + 1} = sprintf('True cap%g [%s]', trueCaps(ci) / 100, dTag); %#ok<SAGROW>
        plannerColors(end + 1, :) = amberRamp(ci, :);   %#ok<SAGROW>
        plannerStyles{end + 1}    = dStyle;             %#ok<SAGROW>
        plannerMarkers{end + 1}   = '^';                %#ok<SAGROW>
        plannerWidths(end + 1)    = 1.6;                %#ok<SAGROW>
        plannerBaseline(end + 1)  = false;              %#ok<SAGROW>
        plannerDeltaIdx(end + 1)  = di;                 %#ok<SAGROW>
    end

    % --- KPAXCap ---
    for ci = 1:numel(kpaxCapCaps)
        if ~ismember(kpaxCapCaps(ci), dCaps), continue; end
        plannerNames{end + 1}   = sprintf('KPAXCap_cap%d', kpaxCapCaps(ci)); %#ok<SAGROW>
        plannerDisplay{end + 1} = sprintf('KPAXCap cap%g [%s]', kpaxCapCaps(ci) / 100, dTag); %#ok<SAGROW>
        plannerColors(end + 1, :) = mossRamp(ci, :);    %#ok<SAGROW>
        plannerStyles{end + 1}    = dStyle;             %#ok<SAGROW>
        plannerMarkers{end + 1}   = 'v';                %#ok<SAGROW>
        plannerWidths(end + 1)    = 2.0;                %#ok<SAGROW>
        plannerBaseline(end + 1)  = true;               %#ok<SAGROW>
        plannerDeltaIdx(end + 1)  = di;                 %#ok<SAGROW>
    end

    % --- Baselines, once per delta, drawn thick so they read as anchors ---
    plannerNames    = [plannerNames,   {'KPAX', 'KinoPaxPlus'}];                          %#ok<AGROW>
    plannerDisplay  = [plannerDisplay, {sprintf('KPAX [%s]', dTag), ...
                                        sprintf('KinoPaxPlus [%s]', dTag)}];              %#ok<AGROW>
    plannerColors   = [plannerColors;  0.10 0.10 0.10;  0.20 0.40 0.80];                  %#ok<AGROW>
    plannerStyles   = [plannerStyles,  {dStyle, dStyle}];                                 %#ok<AGROW>
    plannerMarkers  = [plannerMarkers, {'s', 'd'}];                                       %#ok<AGROW>
    plannerWidths   = [plannerWidths,  2.5, 2.5];                                         %#ok<AGROW>
    plannerBaseline = [plannerBaseline, true, true];                                      %#ok<AGROW>
    plannerDeltaIdx = [plannerDeltaIdx, di, di];                                          %#ok<AGROW>
end

numRunsPer = 50 * ones(1, numel(plannerNames));   % max runs searched (missing files skipped)

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
        fprintf('\n=== Environment: %s | Cost metric: %s ===\n', env, costTitle);

        % --- Load every (planner, delta) series for this cost metric ---
        % Each series builds its own delta_metric token, so the three deltas overlay in one figure.
        R = cell(1, nPlanner);
        for pi = 1:nPlanner
            tok   = sprintf('%s_%s', deltas{plannerDeltaIdx(pi)}, metric);
            R{pi} = loadRuns(dataDir, env, plannerNames{pi}, tok, numRunsPer(pi));
            fprintf('  %-34s %-18s : %d runs\n', plannerNames{pi}, ...
                    ['[' deltas{plannerDeltaIdx(pi)} ']'], numel(R{pi}));
        end

        %% ---------- FIGURE: Best Cost vs Time (mean lines, no bands) ----------
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

        %% ---------- Aggregate summary metrics per planner ----------
        mFirstIter    = NaN(1, nPlanner);
        mFirstSolTime = NaN(1, nPlanner);
        mFirstSolTree = NaN(1, nPlanner);
        mFinalCost    = NaN(1, nPlanner);
        mTotalTime    = NaN(1, nPlanner);
        mSuccess      = NaN(1, nPlanner);
        for pi = 1:nPlanner
            runs = R{pi};
            fiVals = []; fstVals = []; ftsVals = []; fcVals = []; ttVals = [];
            nSol = 0; nTot = 0;
            for ri = 1:numel(runs)
                if isempty(runs{ri}), continue; end
                nTot = nTot + 1;
                fi = firstSolIter(runs{ri}, MAX_FLOAT_THRESH);
                if fi > 0, nSol = nSol + 1; fiVals(end + 1) = fi; end %#ok<SAGROW>
                ft = firstSolTime(runs{ri}, MAX_FLOAT_THRESH);
                if ft >= 0, fstVals(end + 1) = ft; end %#ok<SAGROW>
                fts = firstSolTreeSize(runs{ri}, MAX_FLOAT_THRESH);
                if fts >= 0, ftsVals(end + 1) = fts; end %#ok<SAGROW>
                fc = finalCost(runs{ri}, MAX_FLOAT_THRESH);
                if ~isnan(fc), fcVals(end + 1) = fc; end %#ok<SAGROW>
                ttVals(end + 1) = runs{ri}.elapsed_time_ms(end) / 1000; %#ok<SAGROW>
            end
            if ~isempty(fiVals),  mFirstIter(pi)    = mean(fiVals);  end
            if ~isempty(fstVals), mFirstSolTime(pi) = mean(fstVals); end
            if ~isempty(ftsVals), mFirstSolTree(pi) = mean(ftsVals); end
            if ~isempty(fcVals),  mFinalCost(pi)    = mean(fcVals);  end
            if ~isempty(ttVals),  mTotalTime(pi)    = mean(ttVals);  end
            if nTot > 0,          mSuccess(pi)      = 100 * nSol / nTot; end
        end

        %% ---------- FIGURE: Summary bars (no error bars) ----------
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Summary (%s)', envTitle, costTitle), ...
               'Position', [70 60 1600 880]);

        subplot(2, 3, 1);
        plannerBar(mFirstIter, plannerDisplay, plannerColors, 'Iteration', 'First Solution Iteration');

        subplot(2, 3, 2);
        plannerBar(mFirstSolTime, plannerDisplay, plannerColors, 'Time (ms)', 'Avg Time to First Solution');

        subplot(2, 3, 3);
        plannerBar(mFirstSolTree, plannerDisplay, plannerColors, 'Tree Size (nodes)', 'Avg Tree Size at First Solution');

        subplot(2, 3, 4);
        plannerBar(mFinalCost, plannerDisplay, plannerColors, costYLab, 'Final Best Cost');

        subplot(2, 3, 5);
        plannerBar(mTotalTime, plannerDisplay, plannerColors, 'Time (s)', 'Total Execution Time');

        sgtitle(sprintf('Cost-Prune Tuning Grid at %s \x2014 %s, %s (run means)', ...
                deltaLabel, envTitle, costTitle), 'FontSize', 12, 'FontWeight', 'bold');

        %% ---------- FIGURE: Solution Success Rate ----------
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Success Rate (%s)', envTitle, costTitle), ...
               'Position', [100 100 1300 560]);
        plannerBar(mSuccess, plannerDisplay, plannerColors, 'Success Rate (%)', ...
            sprintf('Solution Success Rate \x2014 %s, %s, %s', envTitle, deltaLabel, costTitle));
        ylim([0 110]);

        %% ---------- FIGURE: Tuning tradeoff scatter ----------
        % Mean time to first solution (x) vs mean final best cost (y). One marker per
        % variant; the lower-left corner is the winning corner (fast AND cheap).
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Tradeoff Scatter (%s)', envTitle, costTitle), ...
               'Position', [130 140 1180 700]);
        hold on;
        for pi = 1:nPlanner
            x = mFirstSolTime(pi);
            y = mFinalCost(pi);
            if isnan(x) || isnan(y), continue; end   % never solved -> nothing to place
            % Explicit flag, not a positional guess: the baselines are no longer simply the
            % last two series (each delta contributes its own baselines and KPAXCap pair).
            if plannerBaseline(pi), msz = 11; else, msz = 8; end
            plot(x, y, plannerMarkers{pi}, ...
                 'MarkerFaceColor', plannerColors(pi, :), ...
                 'MarkerEdgeColor', 'k', 'LineWidth', 0.5, ...
                 'MarkerSize', msz, 'DisplayName', plannerDisplay{pi});
        end
        xlabel('Avg Time to First Solution (ms)'); ylabel(sprintf('Avg Final %s', costYLab));
        grid on;
        clickableLegend();
        title(sprintf(['Tuning Tradeoff: Time to First Solution vs Final Cost \x2014 %s, %s\n' ...
                       'lower-left is better (fast and cheap); ' ...
                       '\x25cb CleanCost grid, \x25b3 TrueStar, \x25bd KPAXCap, \x25a1 baseline'], ...
                       envTitle, costTitle), 'FontWeight', 'bold');
    end   % cost metric loop
end  % environment loop

fprintf('\nAll figures generated (%d total). Click a legend entry to hide/show that series.\n', figNum);

%% ====================== helper functions ======================

function runs = loadRuns(dataDir, env, planner, delta, numRuns)
    % Load one (planner, delta) series' per-run CSVs; missing files are skipped.
    % 'delta' is the full filename token, e.g. 'large_length' or 'fine_control_effort' -- the
    % caller builds it from the series' own delta, which is what lets the three deltas overlay.
    runs = {};
    for ri = 0:(numRuns - 1)
        switch planner
            case 'KinoPaxPlus'
                fn = sprintf('%s_delta%s_run%d.csv', env, delta, ri);
            case 'KPAX'
                fn = sprintf('%s_KPAX_delta%s_run%d.csv', env, delta, ri);
            otherwise
                % STAR variants and KPAXCap use their planner label directly as the filename
                % token. The 'KPAX' arm above is an exact switch case, so it cannot swallow
                % 'KPAXCap_*'.
                if startsWith(planner, 'KinoPaxSTAR') || startsWith(planner, 'KPAXCap')
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

function plotMeanTime(runs, col, commonTime, color, style, width, name)
    % Mean of column 'col' vs a shared time grid (previous-sample hold). No band:
    % with 59 overlaid series the +/-std fills made the figure unreadable.
    %
    % RIGHT-TAIL HOLD (this is the fix for the final-cost reporting bug). commonTime spans
    % globalMaxTime across ALL planners, and interp1(..., 'previous', NaN) returns NaN for every
    % query past a run's last sample. With mean(..., 'omitnan') below, the right-hand tail would
    % then average only the runs that happened to last longest -- so the end of the curve reported
    % the single longest run's cost rather than the mean of each run's final cost. Holding each
    % run's last observed value forward fixes it, and is the correct semantics here: best_cost is
    % monotone non-increasing and a finished run's best cost genuinely IS still that value.
    % With the hold in place the curve's right edge equals the "Final Best Cost" bar exactly.
    %
    % Leading NaNs (before a run's first sample) are left alone -- the run has produced nothing yet.
    % A run that never solved carries the MAX_FLOAT sentinel, which sanitize() turns into NaN after
    % this loop, so holding it forward still correctly excludes it.
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
            row(commonTime > t(end)) = v(end);   % hold the run's final value forward
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

function clickableLegend()
    % Legend whose entries toggle their series on click — the only way 59 overlaid
    % series stay readable. ItemHitFcn needs R2016a+; this repo is on R2023a.
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
    % Drop cost sentinels (MAX_FLOAT / INFINITY) so they don't distort means.
    A(A > 1e30) = NaN;
    A(A == -1)  = NaN;   % harmless for cost/tree_size; guards diagnostic sentinels
end

function plannerBar(mu, plannerLabels, plannerColors, ylab, ttl)
    % One coloured bar per variant. No error bars by request — with 59 bars the
    % whiskers obscured more than they conveyed.
    hold on;
    nP = numel(mu);
    for pi = 1:nP
        bar(pi, mu(pi), 0.7, 'FaceColor', plannerColors(pi, :), 'EdgeColor', 'none');
    end
    set(gca, 'XTick', 1:nP, 'XTickLabel', plannerLabels, 'FontSize', 6);
    xtickangle(50);
    xlim([0.5, nP + 0.5]);
    ylabel(ylab); title(ttl); grid on;
end

function it = firstSolIter(tbl, thresh)
    % Iteration at which this run first reached a finite (< thresh) best_cost; -1 if never.
    solIdx = find(tbl.best_cost < thresh, 1, 'first');
    if isempty(solIdx), it = -1; else, it = tbl.iteration(solIdx); end
end

function t = firstSolTime(tbl, thresh)
    % Elapsed time (ms) at which this run first reached a finite (< thresh) best_cost;
    % -1 if it never found a solution.
    solIdx = find(tbl.best_cost < thresh, 1, 'first');
    if isempty(solIdx), t = -1; else, t = tbl.elapsed_time_ms(solIdx); end
end

function n = firstSolTreeSize(tbl, thresh)
    % Tree size at the iteration this run first reached a finite (< thresh) best_cost;
    % -1 if it never found a solution.
    solIdx = find(tbl.best_cost < thresh, 1, 'first');
    if isempty(solIdx), n = -1; else, n = tbl.tree_size(solIdx); end
end

function c = finalCost(tbl, thresh)
    % Last finite best_cost in one run; NaN if the run never found a solution.
    costs = tbl.best_cost;
    costs(costs > thresh) = NaN;
    v = costs(~isnan(costs));
    if isempty(v), c = NaN; else, c = v(end); end
end
