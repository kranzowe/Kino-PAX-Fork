%% Cost Tuning Sweep Visualization — cap x exp grid, both cost metrics
% Reads per-iteration CSVs produced by kinopaxstar_cost_tuning_sweep.cu
% (run via run_cost_tuning_sweep.sh) and compares the full cost-prune tuning grid
% at the single Large delta (R1 = 27k regions).
%
% Grid: KinoPaxSTARWeightedCost over w {0, 0.25, 0.5, 0.75, 1.0} x k {0.5, 1, 2} x ancestor
% {node-only, chain} = 30 variants, plus 2 cost-prune union-blend reference probes, plus
% KinoPaxSTARNoPruneAncestor in all three ancestor-pruning modes, plus KPAX and KinoPaxPlus
% as reference baselines. 37 series.
%
% WeightedCost replaces costprune's MULTIPLICATIVE blend (costProb * syclop) with a weighted sum
%   P_combined = min(1, w*P_syclop + (1-w)*P_cost + P_floor)
% applied at both acceptance points, where P_syclop = vertexScore + fAccept is the full KPAX rule
% and P_floor is fixed at EPSILON. w = 1 reproduces KPAX's acceptance, w = 0 is pure cost-greedy,
% so the w axis brackets the exploration/cost tradeoff with one knob. P_cost is
% exp(-k*(cost-m)/(mean-m)): exactly 1 at the region min AND with a real gradient across the whole
% range, unlike min(1,(mean/cost)^k), which is pinned at 1 for every cost at or below the mean.
%
% KinoPaxSTARNoPruneAncestor is KinoPaxSTARNoPrune with KinoPaxPlus's retroactive ancestor
% pruning. The NoPrune base matters: it reactivates with the plain KPAX rule
% `vertexScores + fAccept` (ADDITIVE), so it keeps KPAX-equivalent exploration in cluttered
% maps and ancestor pruning is the only variable.
%   off    h_ancestorPrune_ = 0  -- reproduces stock KinoPaxSTARNoPrune exactly (control arm)
%   node   h_ancestorPrune_ = 1  -- prune a node beaten in its own region
%   chain  h_ancestorPrune_ = 2  -- prune if any ancestor is beaten (memoized, O(1))
%
% The union probes swap CostPrune's reactivation from `costProb * syclop` to
% `fmaxf(costProb, syclop)` with floor 0, restoring the additive fAccept floor that the
% product form destroys. That is the direct test of why CostPrune trails KPAX in the zigzag.
%
% COST METRIC: swept by rebuilding, since COST_MODE is a compile-time #if inside
% edgeCost (include/helper/helper.cuh). The metric therefore rides in the delta
% token of every filename: large_effort (control effort) vs large_length (workspace
% path length). The two metrics have DIFFERENT UNITS, so every cost-dependent
% figure below is drawn once per metric and they are never overlaid.
%
% Per-run CSVs (in dataDir):
%   KinoPaxPlus:  {env}_delta{large_METRIC}_run{n}.csv
%   KPAX:         {env}_KPAX_delta{large_METRIC}_run{n}.csv
%   ancestor:     {env}_KinoPaxSTARNoPruneAncestor_{off,node,chain}_delta{large_METRIC}_run{n}.csv
%   union probes: {env}_KinoPaxSTARcostprune_union_cap{N}_exp50_floor0_delta{large_METRIC}_run{n}.csv
%   weighted:     {env}_KinoPaxSTARWeightedCost_w{N}_k{N}_anc{N}_delta{large_METRIC}_run{n}.csv
%
% Per-iteration columns: iteration, frontier_size, tree_size, elapsed_time_ms, best_cost
%
% ENCODING: the WeightedCost grid needs three dimensions, so hue family = ancestor mode
% (steel-blue for anc1 = node-only, purple for anc2 = memoized chain), shade = w (light->dark),
% and line style = k (':' 0.5, '--' 1.0, '-' 2.0). The three NoPruneAncestor modes are an
% orange->dark-red ramp and the two union probes are a green pair, both drawn thick. Baselines
% are near-black (KPAX) and DASHED blue (KinoPaxPlus, dashed so it does not read as part of the
% steel-blue weighted family). Every legend here is CLICKABLE — click an entry to hide/show that
% series (see toggleSeries).
%
% FAIR-COMPARISON NOTE: an "iteration" is a different unit of work per planner, so
% cost-vs-TIME is the fair cross-planner axis. Error bands and error bars are
% deliberately omitted throughout; the scatter shows run means only.
%
% USAGE: cd into the data directory, then call the script BY NAME, not via run():
%   cd build/Data/Benchmarks/KinoPaxStarCostTuning
%   addpath('<repo>/scripts')
%   process_cost_tuning_and_plot
% run('<abs path>/process_cost_tuning_and_plot.m') would cd to the scripts folder
% first, and dataDir below ('' = current folder) would then find nothing.

clear; clc; close all;

%% --- Configuration ---
dataDir = '';   % '' = current directory (run this from Data/Benchmarks/KinoPaxStarCostTuning)

environments = {'zigzag'};
envTitles    = {'Zigzag Corridor'};

% Cost metric axis — one build each, so one set of figures each.
costTokens  = {'large_effort', 'large_length'};
costTitles  = {'Control Effort', 'Workspace Path Length'};
costYLabels = {'Path Cost (control effort)', 'Path Cost (workspace path length)'};

deltaLabel = 'Large-\delta (27k)';

% WeightedCost grid — must match WEIGHTS / WEIGHTED_EXPS / WEIGHTED_ANC in
% kinopaxstar_cost_tuning_sweep.cu. Values are the integer label tokens (100 x the float for
% w and k), exactly as they appear in the filenames.
weights   = [0 25 50 75 100];
wExps     = [50 100 200];
wAnc      = [1 2];         % 1 = node-only, 2 = memoized chain (mode 0/off dropped)

% Hue family = ancestor mode (steel = node-only, purple = chain), shade = w, line style = k.
steelRamp  = [0.72 0.83 0.93;    % w 0    - steel blue (lightest)
              0.48 0.66 0.85;    % w 0.25
              0.26 0.48 0.72;    % w 0.5
              0.13 0.32 0.55;    % w 0.75
              0.04 0.17 0.34];   % w 1.0  - steel blue (darkest)
purpleRamp = [0.85 0.76 0.93;    % w 0    - purple (lightest)
              0.68 0.52 0.84;    % w 0.25
              0.51 0.31 0.71;    % w 0.5
              0.36 0.16 0.54;    % w 0.75
              0.21 0.05 0.35];   % w 1.0  - purple (darkest)
wExpStyles = {':', '--', '-'};   % k 0.5, 1.0, 2.0
ancRamps   = {steelRamp, purpleRamp};
ancMarkers = {'o', 'p'};

% NoPruneAncestor modes: their own orange->dark-red ramp.
ancestorTokens = {'off', 'node', 'chain'};
ancestorLabels = {'Anc off (=NoPrune)', 'Anc node-only', 'Anc chain'};
ancestorRamp   = [0.98 0.72 0.35;    % off   - light orange
                  0.90 0.42 0.10;    % node  - orange
                  0.60 0.12 0.08];   % chain - dark red

% Union-blend probes: fmaxf(costProb, syclop) with floor 0, kept as the reference the
% weighted sum is meant to beat. Green so they never read as part of another family.
unionNames  = {'KinoPaxSTARcostprune_union_cap0_exp50_floor0', ...
               'KinoPaxSTARcostprune_union_cap100_exp50_floor0'};
unionLabels = {'Union cap0', 'Union cap1'};
unionRamp   = [0.55 0.85 0.45;    % cap 0   - light green
               0.13 0.52 0.20];   % cap 1.0 - dark green

% --- Build the series arrays (30 weighted + 2 union + 3 ancestor + 2 baselines) ---
plannerNames   = {};
plannerDisplay = {};
plannerColors  = [];
plannerStyles  = {};
plannerMarkers = {};
plannerWidths  = [];
for ai = 1:numel(wAnc)
    for wi = 1:numel(weights)
        for ei = 1:numel(wExps)
            plannerNames{end + 1}   = sprintf('KinoPaxSTARWeightedCost_w%d_k%d_anc%d', ...
                                              weights(wi), wExps(ei), wAnc(ai)); %#ok<SAGROW>
            plannerDisplay{end + 1} = sprintf('W w%g k%g a%d', weights(wi) / 100, ...
                                              wExps(ei) / 100, wAnc(ai)); %#ok<SAGROW>
            plannerColors(end + 1, :) = ancRamps{ai}(wi, :);   %#ok<SAGROW>
            plannerStyles{end + 1}    = wExpStyles{ei};        %#ok<SAGROW>
            plannerMarkers{end + 1}   = ancMarkers{ai};        %#ok<SAGROW>
            plannerWidths(end + 1)    = 1.4;                   %#ok<SAGROW>
        end
    end
end
for ui = 1:numel(unionNames)
    plannerNames{end + 1}   = unionNames{ui};   %#ok<SAGROW>
    plannerDisplay{end + 1} = unionLabels{ui};  %#ok<SAGROW>
    plannerColors(end + 1, :) = unionRamp(ui, :);  %#ok<SAGROW>
    plannerStyles{end + 1}    = '-';               %#ok<SAGROW>
    plannerMarkers{end + 1}   = 'v';               %#ok<SAGROW>
    plannerWidths(end + 1)    = 2.2;               %#ok<SAGROW>
end
for ai = 1:numel(ancestorTokens)
    plannerNames{end + 1}   = sprintf('KinoPaxSTARNoPruneAncestor_%s', ancestorTokens{ai}); %#ok<SAGROW>
    plannerDisplay{end + 1} = ancestorLabels{ai};        %#ok<SAGROW>
    plannerColors(end + 1, :) = ancestorRamp(ai, :);     %#ok<SAGROW>
    plannerStyles{end + 1}    = '-';                     %#ok<SAGROW>
    plannerMarkers{end + 1}   = '^';                     %#ok<SAGROW>
    plannerWidths(end + 1)    = 2.2;                     %#ok<SAGROW>
end
% Reference baselines, drawn thick so they read as anchors
plannerNames   = [plannerNames,   {'KPAX', 'KinoPaxPlus'}];
plannerDisplay = [plannerDisplay, {'KPAX', 'KinoPaxPlus'}];
plannerColors  = [plannerColors;  0.10 0.10 0.10;  0.20 0.40 0.80];
plannerStyles  = [plannerStyles,  {'-', '--'}];   % KinoPaxPlus dashed: steel-blue is taken
plannerMarkers = [plannerMarkers, {'s', 's'}];
plannerWidths  = [plannerWidths,  2.5, 2.5];

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
                       '\x25cb weighted anc0, \x2606 weighted anc2, \x25bd union, \x25b3 NoPruneAnc, \x25a1 baseline'], ...
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
            case 'KPAX'
                fn = sprintf('%s_KPAX_delta%s_run%d.csv', env, delta, ri);
            otherwise
                % The WeightedCost grid, the union probes and the NoPruneAncestor modes
                % all use the planner name directly as the filename token.
                if startsWith(planner, 'KinoPaxSTARcostprune') || ...
                   startsWith(planner, 'KinoPaxSTARNoPruneAncestor') || ...
                   startsWith(planner, 'KinoPaxSTARWeightedCost')
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
    % with 26 overlaid series the +/-std fills made the figure unreadable.
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
    set(gca, 'XTick', 1:nP, 'XTickLabel', plannerLabels, 'FontSize', 5);
    xtickangle(60);
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
