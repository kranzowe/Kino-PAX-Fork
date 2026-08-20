%% Algorithm Comparison — time to first solution vs final cost
% Reads per-iteration CSVs produced by kinopaxstar_comparison.cu (run via
% run_comparison_benchmark.sh) and draws one scatter per cost metric:
%
%   x = mean time to first solution (ms)     "how fast does it find ANYTHING"
%   y = mean final best cost                 "how good is what it ends with"
%
% Lower-left wins. Points are labelled in place rather than by legend — with 8 algorithms
% that reads far better.
%
% ALGORITHMS
%   KPAX, KinoPaxPlus                baselines
%   KinoPaxSTARNoGoalBias            no cost pruning
%   KinoPaxSTARTrue                  guarded stale-best pruning
%   KinoPaxSTARTrueAnc               guarded stale-best + ancestor chain
%   KinoPaxSTARWeightedCost          w = 0.9, k = 1, no cost pruning
%   KinoPaxSTARTrueWeightedCost      w = 0.9, k = 1, guarded stale-best
%   KinoPaxSTARTrueWeightedCostAnc   w = 0.9, k = 1, guarded chain
%
% "Guarded" = cost pruning only touches nodes admitted BECAUSE they were their region's
% minimum; Syclop-admitted explorers are never pruned. None of these carry goal-bias
% acceptance — that lives only in KinoPaxSTAR.
%
% ONE ENVIRONMENT PER RUN. The benchmark writes each environment to its own subfolder
% (Data/Benchmarks/KinoPaxStarComparison/<env>/), so cd into the one you want and set
% envName below to match.
%
% USAGE: cd into the data directory, then call this script BY NAME, not via run():
%   cd build/Data/Benchmarks/KinoPaxStarComparison/zigzag     % or .../house
%   addpath('<repo>/scripts')
%   plot_comparison_scatter
% run('<abs path>/plot_comparison_scatter.m') would cd to the scripts folder first, and
% dataDir below ('' = current folder) would then find nothing.

clear; clc; close all;

%% --- Configuration ---
dataDir = '';   % '' = current directory

% Must match the subfolder you cd'd into: 'zigzag' or 'house'.
envName  = 'zigzag';
envTitle = 'Zigzag Corridor';

% Cost metric axis — one build each, so one figure each. The two have DIFFERENT UNITS and
% are never overlaid.
costTokens  = {'large_effort', 'large_length'};
costTitles  = {'Control Effort', 'Workspace Path Length'};
costYLabels = {'Final Path Cost (control effort)', 'Final Path Cost (workspace path length)'};

deltaLabel = 'Large-\delta (27k)';

% Baselines near-black/blue; the un-pruned variants light; the pruned ones progressively
% darker within their family (steel = plain STAR, magenta = weighted).
plannerNames = { ...
    'KPAX', ...
    'KinoPaxPlus', ...
    'KinoPaxSTARNoGoalBias', ...
    'KinoPaxSTARTrue', ...
    'KinoPaxSTARTrueAnc', ...
    'KinoPaxSTARWeightedCost', ...
    'KinoPaxSTARTrueWeightedCost', ...
    'KinoPaxSTARTrueWeightedCostAnc'};
plannerDisplay = { ...
    'KPAX', 'KinoPaxPlus', 'NoGoalBias', 'True', 'TrueAnc', ...
    'WCost', 'TrueWCost', 'TrueWCostAnc'};
plannerColors = [ ...
    0.10 0.10 0.10;    % KPAX                 - near-black
    0.20 0.40 0.80;    % KinoPaxPlus          - blue
    0.62 0.76 0.90;    % NoGoalBias           - steel (light)
    0.26 0.48 0.72;    % True                 - steel (mid)
    0.06 0.22 0.42;    % TrueAnc              - steel (dark)
    0.95 0.72 0.88;    % WCost                - magenta (light)
    0.80 0.30 0.62;    % TrueWCost            - magenta (mid)
    0.42 0.06 0.30];   % TrueWCostAnc         - magenta (dark)
plannerMarkers = {'s', 's', 'o', 'o', 'o', 'd', 'd', 'd'};

numRunsSearched  = 50;     % max runs looked for per planner; missing files skipped
MAX_FLOAT_THRESH = 1e30;   % best_cost sentinel (MAX_FLOAT / INFINITY) -> NaN

nPlanner = numel(plannerNames);

%% ======================================================================
for mi = 1:numel(costTokens)
    costTok   = costTokens{mi};
    costTitle = costTitles{mi};
    costYLab  = costYLabels{mi};
    fprintf('\n=== %s | %s ===\n', envName, costTitle);

    mFirstSolTime = NaN(1, nPlanner);
    mFinalCost    = NaN(1, nPlanner);
    nLoaded       = zeros(1, nPlanner);

    for pi = 1:nPlanner
        runs = loadRuns(dataDir, envName, plannerNames{pi}, costTok, numRunsSearched);
        nLoaded(pi) = numel(runs);

        fstVals = [];
        fcVals  = [];
        for ri = 1:numel(runs)
            if isempty(runs{ri}), continue; end
            t = firstSolTime(runs{ri}, MAX_FLOAT_THRESH);
            if t >= 0, fstVals(end + 1) = t; end %#ok<SAGROW>
            c = finalCost(runs{ri}, MAX_FLOAT_THRESH);
            if ~isnan(c), fcVals(end + 1) = c; end %#ok<SAGROW>
        end
        if ~isempty(fstVals), mFirstSolTime(pi) = mean(fstVals); end
        if ~isempty(fcVals),  mFinalCost(pi)    = mean(fcVals);  end

        fprintf('  %-34s : %2d runs, t_first = %8.1f ms, final = %.4f\n', ...
                plannerNames{pi}, nLoaded(pi), mFirstSolTime(pi), mFinalCost(pi));
    end

    %% ---------- the scatter ----------
    figure('Name', sprintf('%s - Comparison (%s)', envTitle, costTitle), ...
           'Position', [60 60 900 680], 'Color', 'w');
    hold on;

    for pi = 1:nPlanner
        x = mFirstSolTime(pi);
        y = mFinalCost(pi);
        if isnan(x) || isnan(y)
            fprintf('  (skipping %s: never solved or no data)\n', plannerNames{pi});
            continue;
        end
        plot(x, y, plannerMarkers{pi}, 'MarkerSize', 12, ...
             'MarkerFaceColor', plannerColors(pi, :), 'MarkerEdgeColor', 'k', 'LineWidth', 0.75);
        text(x, y, ['  ' plannerDisplay{pi}], 'FontSize', 9, ...
             'VerticalAlignment', 'middle', 'HorizontalAlignment', 'left');
    end

    % Breathing room on the right so the in-place labels are not clipped by the axes box.
    xl = xlim; xlim([xl(1), xl(1) + 1.28 * (xl(2) - xl(1))]);

    grid on; box on;
    xlabel('Mean Time to First Solution (ms)');
    ylabel(sprintf('Mean %s', costYLab));
    title(sprintf(['Algorithm Comparison \x2014 %s, %s, %s\n' ...
                   'lower-left is better (fast to a first solution, and cheap at the end)'], ...
                   envTitle, deltaLabel, costTitle), 'FontWeight', 'bold');
end

fprintf('\nDone. One figure per cost metric.\n');

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
                % Every STAR variant uses its label directly as the filename token.
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

function t = firstSolTime(tbl, thresh)
    % Elapsed time (ms) at which this run first reached a finite (< thresh) best_cost;
    % -1 if it never found a solution.
    solIdx = find(tbl.best_cost < thresh, 1, 'first');
    if isempty(solIdx), t = -1; else, t = tbl.elapsed_time_ms(solIdx); end
end

function c = finalCost(tbl, thresh)
    % Last finite best_cost in one run; NaN if the run never found a solution.
    costs = tbl.best_cost;
    costs(costs > thresh) = NaN;
    v = costs(~isnan(costs));
    if isempty(v), c = NaN; else, c = v(end); end
end
