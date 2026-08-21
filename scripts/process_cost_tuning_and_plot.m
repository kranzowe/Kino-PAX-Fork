%% Cost Tuning Sweep Visualization — w x k x cap grid, both cost metrics
% Reads per-iteration CSVs produced by kinopaxstar_cost_tuning_sweep.cu
% (run via run_cost_tuning_sweep.sh).
%
% Series (43 total):
%   KinoPaxSTARCleanCost  w {0.2, 0.5, 0.8, 1.0} x k {0.5, 1, 2} x cap {0.25, 0.5, 1.0} = 36
%   KinoPaxSTARTrue       cap {0.1, 0.25, 0.5, 1.0}                                     =  4
%   KPAX, KinoPaxPlus, KinoPaxPlus (fine)                                               =  3
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
%                        e.g. KinoPaxSTARCleanCost_w50_k100_cap25, KinoPaxSTARTrue_cap25
%
% Per-iteration columns: iteration, frontier_size, tree_size, elapsed_time_ms, best_cost
%
% ENCODING (three axes, so three visual channels): colour = w (steel-blue ramp, light->dark as
% w goes 0.2->1.0), line style = k (':' 0.5, '-.' 1.0, '--' 2.0), line width = cap (thin 0.25,
% medium 0.5, thick 1.0). TrueStar is a separate warm/orange ramp over its cap axis, solid.
% Baselines are near-black (KPAX), DASHED blue (KinoPaxPlus) and DOTTED blue (KinoPaxPlus fine) --
% dashed/dotted so they do not read as part of the steel-blue ramp. Every legend here is
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
costTokens  = {'large_effort', 'large_length'};
costTitles  = {'Control Effort', 'Workspace Path Length'};
costYLabels = {'Path Cost (control effort)', 'Path Cost (workspace path length)'};

deltaLabel = 'Large-\delta (27k)';

% CleanCost grid — must match WEIGHTS / WEIGHTED_EXPS / CAPS in
% kinopaxstar_cost_tuning_sweep.cu. Values are the integer label tokens (100 x the float),
% exactly as they appear in the filenames.
weights = [20 50 80 100];
wExps   = [50 100 200];
caps    = [25 50 100];

% TrueStar cap sweep — must match TRUE_CAPS in the benchmark.
trueCaps = [10 25 50 100];

% Colour = w (light->dark), line style = k, line width = cap.
steelRamp  = [0.62 0.76 0.90;    % w 0.2 - steel blue (lightest)
              0.36 0.57 0.79;    % w 0.5
              0.16 0.35 0.58;    % w 0.8
              0.03 0.15 0.31];   % w 1.0 - steel blue (darkest)
wExpStyles = {':', '-.', '--'};        % k 0.5, 1.0, 2.0
capWidths  = [0.8, 1.4, 2.2];          % cap 0.25, 0.5, 1.0

% TrueStar gets its own warm ramp so it never reads as part of the CleanCost grid.
amberRamp = [0.99 0.80 0.54;     % cap 0.10 (lightest)
             0.96 0.63 0.26;
             0.85 0.44 0.09;
             0.60 0.28 0.02];    % cap 1.00 (darkest)

% --- Build the series arrays (36 CleanCost + 4 TrueStar + 3 baselines) ---
plannerNames   = {};
plannerDisplay = {};
plannerColors  = [];
plannerStyles  = {};
plannerMarkers = {};
plannerWidths  = [];
for wi = 1:numel(weights)
    for ei = 1:numel(wExps)
        for ci = 1:numel(caps)
            plannerNames{end + 1}   = sprintf('KinoPaxSTARCleanCost_w%d_k%d_cap%d', ...
                                              weights(wi), wExps(ei), caps(ci)); %#ok<SAGROW>
            plannerDisplay{end + 1} = sprintf('C w%g k%g cap%g', weights(wi) / 100, ...
                                              wExps(ei) / 100, caps(ci) / 100); %#ok<SAGROW>
            plannerColors(end + 1, :) = steelRamp(wi, :);   %#ok<SAGROW>
            plannerStyles{end + 1}    = wExpStyles{ei};     %#ok<SAGROW>
            plannerMarkers{end + 1}   = 'o';                %#ok<SAGROW>
            plannerWidths(end + 1)    = capWidths(ci);      %#ok<SAGROW>
        end
    end
end
for ci = 1:numel(trueCaps)
    plannerNames{end + 1}   = sprintf('KinoPaxSTARTrue_cap%d', trueCaps(ci)); %#ok<SAGROW>
    plannerDisplay{end + 1} = sprintf('True cap%g', trueCaps(ci) / 100);      %#ok<SAGROW>
    plannerColors(end + 1, :) = amberRamp(ci, :);   %#ok<SAGROW>
    plannerStyles{end + 1}    = '-';                %#ok<SAGROW>
    plannerMarkers{end + 1}   = '^';                %#ok<SAGROW>
    plannerWidths(end + 1)    = 1.6;                %#ok<SAGROW>
end
% Reference baselines, drawn thick so they read as anchors.
% 'KinoPaxPlusFine' is not a planner name in the benchmark -- it is a pseudo-name this script maps
% to the fine-delta KinoPaxPlus CSVs in loadRuns().
plannerNames   = [plannerNames,   {'KPAX', 'KinoPaxPlus', 'KinoPaxPlusFine'}];
plannerDisplay = [plannerDisplay, {'KPAX', 'KinoPaxPlus (27k)', 'KinoPaxPlus (216k)'}];
plannerColors  = [plannerColors;  0.10 0.10 0.10;  0.20 0.40 0.80;  0.20 0.40 0.80];
plannerStyles  = [plannerStyles,  {'-', '--', ':'}];   % steel-blue solid is taken by the grid
plannerMarkers = [plannerMarkers, {'s', 's', 'd'}];
plannerWidths  = [plannerWidths,  2.5, 2.5, 2.5];

numRunsPer = 50 * ones(1, numel(plannerNames));   % max runs searched (missing files skipped)

MAX_FLOAT_THRESH = 1e30;   % best_cost sentinel (MAX_FLOAT / INFINITY) -> NaN
numTimeSamples   = 500;

nPlanner = numel(plannerNames);

%% ======================================================================
figNum = 0;
for ei = 1:numel(environments)
    env      = environments{ei};
    envTitle = envTitles{ei};

    for mi = 1:numel(costTokens)
        costTok   = costTokens{mi};
        costTitle = costTitles{mi};
        costYLab  = costYLabels{mi};
        fprintf('\n=== Environment: %s | Cost metric: %s ===\n', env, costTitle);

        % --- Load every planner's runs for this cost metric ---
        R = cell(1, nPlanner);
        for pi = 1:nPlanner
            R{pi} = loadRuns(dataDir, env, plannerNames{pi}, costTok, numRunsPer(pi));
            fprintf('  %-44s : %d runs\n', plannerNames{pi}, numel(R{pi}));
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
            isBaseline = pi > nPlanner - 2;
            if isBaseline, msz = 11; else, msz = 8; end
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
                       '\x25cb weighted grid, \x25a1 baseline'], ...
                       envTitle, costTitle), 'FontWeight', 'bold');
    end   % cost metric loop
end  % environment loop

fprintf('\nAll figures generated (%d total). Click a legend entry to hide/show that series.\n', figNum);

%% ====================== helper functions ======================

function runs = loadRuns(dataDir, env, planner, delta, numRuns)
    % Load a planner's per-run CSVs for one cost metric; missing files are skipped.
    runs = {};
    for ri = 0:(numRuns - 1)
        switch planner
            case 'KinoPaxPlus'
                fn = sprintf('%s_delta%s_run%d.csv', env, delta, ri);
            case 'KinoPaxPlusFine'
                % Pseudo-name: the finer discretization is a separate binary that writes
                % KinoPaxPlus rows under the 'fine_*' delta token instead of 'large_*'.
                fn = sprintf('%s_delta%s_run%d.csv', env, strrep(delta, 'large', 'fine'), ri);
            case 'KPAX'
                fn = sprintf('%s_KPAX_delta%s_run%d.csv', env, delta, ri);
            otherwise
                % Every STAR variant uses its planner label directly as the filename token.
                if startsWith(planner, 'KinoPaxSTAR')
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
    % with 43 overlaid series the +/-std fills made the figure unreadable.
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
    A  = sanitize(A);
    mu = mean(A, 1, 'omitnan');
    valid = ~isnan(mu);
    if ~any(valid), return; end
    plot(commonTime(valid), mu(valid), style, 'Color', color, ...
         'LineWidth', width, 'DisplayName', name);
end

function clickableLegend()
    % Legend whose entries toggle their series on click — the only way 26 overlaid
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
    % One coloured bar per variant. No error bars by request — with 26 bars the
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
