%% Cost Tuning Sweep Visualization — w x k x cap grid, both cost metrics
% Reads per-iteration CSVs produced by kinopaxstar_cost_tuning_sweep.cu
% (run via run_combo_tuning_sweep.sh).
%
% Series are (planner, delta) pairs; the delta machinery is intact but THIS PASS runs the coarse
% delta and the house environment only (see the commented full sets below and in
% run_combo_tuning_sweep.sh):
%
%   KinoPaxSTARCleanCost  r2 OFF (fixed) x w {0.9, 0.95, 1.0} x k {0.25, 1, 16}
%                         x cap {0.03, 0.1, 1.0}                                  = 21
%   KinoPaxSTARTrue       cap {0.03, 0.1}                                         =  2
%   KPAXCap               cap {0.03, 0.1}                                         =  2
%   KPAX, KinoPaxPlus                                                             =  2
%                                                                                 -----
%                                                                                    27
%
% 21 and not the plain 3*3*3 = 27: at w = 1 the cost term vanishes from weightedAccept, so only
% k = 1 is run there -- the other six points would be the same rule differing only by RNG stream.
%
% r2 = the R2 SUB-REGION SEEDING FREE PASS, now FIXED OFF. With it on (KPAX's behaviour) a candidate
% claiming a virgin R2 sub-region is admitted unconditionally, bypassing the weighted roll; off, it
% takes the same roll as everything else (the KinoPaxSTARnoseed condition). Both arms were measured
% and off is now permanent, so admission is steered only by the Syclop score and the cost term.
% Propagate still marks activeSubVertices, so r2_coverage_pct stays valid and comparable with the
% earlier two-arm data.
%
% TWO NORMALIZATION FIXES land in this data, which is why k and cap are both re-opened:
%   * Graph's Syclop floor is now 1/N_active (the mean share) rather than a fixed EPSILON = 1e-2,
%     which exceeded the score it floored by ~270x and capped the number of discriminated regions
%     at 1/EPSILON = 100 at ANY grid size. OPT-IN: KPAXCap / TrueStar / CleanCost take it; KPAX
%     deliberately keeps the legacy floor, so it stays an unmodified baseline. The score_floor
%     column makes that split visible: flat 0.01 for KPAX, decaying for the others.
%   * CleanCost drops P_floor and uses a GLOBAL cost scale (costProbExpGlobal): the region's own
%     minimum stays the reference, but the denominator is global, so a cost excess means the same
%     thing everywhere instead of being pinned at x ~ 1 in every region by construction. The
%     cost_scale column logs that denominator.
%
% ENCODING: colour = cap (steel ramp light->dark 0.03->1.0 for CleanCost, amber for TrueStar,
% grey-green for KPAXCap, near-black KPAX, blue KinoPaxPlus); line style = k (':' 0.25, '-' 1, '--' 16);
% marker = w ('o' 0.9, 'x' 0.95, '+' 1.0, scatter only); line width = delta (constant with one
% delta and one r2 arm; it separates them again as soon as either set is restored). Every legend
% here is CLICKABLE — click an entry to hide/show that series.
%
% FAIR-COMPARISON NOTE: an "iteration" is a different unit of work per planner, so
% cost-vs-TIME is the fair cross-planner axis. Error bands and error bars are
% deliberately omitted throughout; the scatter shows run means only.
%
% USAGE: cd into the data directory, then call the script BY NAME, not via run():
%   cd build/Data/Benchmarks/CountingStars/zigzag     % or .../narrowPassage
%   addpath('<repo>/scripts')
%   process_countingstars_and_plot
% run('<abs path>/process_countingstars_and_plot.m') would cd to the scripts folder
% first, and dataDir below ('' = current folder) would then find nothing.

clear; clc; close all;

%% --- Configuration ---
dataDir = '';   % '' = current directory (run this from Data/Benchmarks/CountingStars/<env>)

% One environment per run — must match the subfolder you cd'd into.
%   'zigzag' -> 'Zigzag Corridor',  'house' -> 'House'
% SCOPE: zigzag and narrowPassage this pass (matches ENV_NAMES in run_combo_tuning_sweep.sh).
% ONE PER RUN -- each environment writes to its own subfolder, so set this to match the folder you
% cd'd into and re-run for the other.
%   'zigzag' -> 'Zigzag Corridor',  'narrowPassage' -> 'Narrow Passage',  'house' -> 'House'
environments = {'house'};
envTitles    = {'House'};
% environments = {'narrowPassage'};   envTitles = {'Narrow Passage'};

% Cost metric axis — one build each, so one set of figures each.
metrics      = {'length'};
metricTitles = {'Workspace Path Length'};
metricYLabels = {'Path Cost (workspace path length)'};

% Delta axis — OVERLAID inside each figure, encoded as line WIDTH. The filename token is
% sprintf('%s_%s', delta, metric), e.g. 'fine_control_length'.
deltas      = {'large', 'fine', 'tiny'};
deltaTitles = {'27k', '216k W-refined', '593k V-refined'};
deltaWidths = [1.0, 1.8, 2.6];

% WHICH ARMS EXIST AT EACH DELTA. Index 0 runs the full sweep; the two finer deltas run
% KINOPAXPLUS ONLY, because KinoPaxPlus is the planner whose whole advantage is a tiny frontier at
% a fine discretisation, so it is the one baseline that has to be measured at all three. Re-running
% the CountingStars grid there would triple the sweep to answer a question the coarse delta already
% answers.
%
% MUST MATCH DELTA_EXTRA_ARGS in run_countingstars_sweep.sh: "--only-kinopaxplus" there is a true
% here. When these drift, loadRuns() silently finds no files and reports "0 runs" for the orphaned
% series rather than erroring -- the failure mode that wastes a whole sweep.
% cross_check_countingstars_grid.py asserts it.
deltaPlusOnly = [false, true, true];

capDerived     = 3;       % label token for cap = 0.03 (CAP_DERIVED in the benchmark)
% --single-point is not used by this sweep, so any delta that runs an arm runs its full axis.
deltaSingleCap = [false, false, false];

deltaLabel = '3 deltas overlaid';

% CountingStars v2 grid - must match GOAL_FRONTIER_SIZES / EXPLORE_FRACS in
% countingstars_sweep.cu. Values are the label tokens exactly as they appear in the filenames:
% goal_frontier_size plain, explore_frac as round(100 x float).
% cross_check_countingstars_grid.py asserts these stay in step with the .cu and the .sh; when they
% drift, MATLAB reports "0 runs" for the orphaned series rather than erroring, which is the failure
% mode that silently wastes a whole sweep.
%
% goal_frontier_size B IS THE HEADLINE AXIS, and it is a different object from v1's react_count.
% react_count was a CAP that F happened to land under; B is the TARGET the doors fill in priority
% order -- optimal (uncapped), then explore_frac of what is left to the freshest regions, then the
% region-best guarantee, then a uniform draw. F is met by construction, so B is the input and
% propagations-per-node is the output. KinoPaxPlus divides the whole budget over a frontier its
% pruning keeps tiny (bf = MAX_TREE_SIZE/(F*32), 40,000 propagations per node at F = 10), which is
% the number prop_attempted/frontier_size is read against.
%
% B = 2000 and 10000 are BELOW NUM_R1_REGIONS (27,000 at the coarse delta), and the optimal door is
% uncapped, so at those points the budget is a SOFT target and budget_used may run over B. That is
% deliberate: it is the direct read on how much of the frontier the optimal door alone accounts for.
csGoalFrontierSizes = [2000 10000 50000];
csExploreFracs      = [5 10 25];

% The derived operating point that --single-point selects.
csDerivedGoalFrontier = 10000;
csDerivedExploreFrac  = 10;

% CleanCost baseline point - one series, the well-tuned operating point. Same label format as the
% cost sweep, so its historical CSVs load here unchanged.
cleanBaseR2  = 'off';
cleanBaseW   = 90;
cleanBaseK   = 100;
cleanBaseCap = 3;

% TrueStar and KPAXCap cap sweeps - must match TRUE_CAPS / KPAXCAP_CAPS in the benchmark.
kpaxCapCaps = [3];

% TWO axes, so they get the two channels the eye reads first and nothing has to double up.
% colour = goal_frontier_size, because B is what the whole design turns on; DARKER IS A SMALLER
% BUDGET, which is the direction the design is pushing. line style = explore_frac.
%   rows: B 2000, B 10000, B 50000
budgetColors = [0.10 0.10 0.10;    % B 2000   (smallest frontier, most propagations per node)
                0.16 0.38 0.63;    % B 10000
                0.62 0.55 0.72];   % B 50000  (largest frontier, fewest propagations per node)
fracStyles   = {'-', '--', ':'};   % explore_frac = 0.05, 0.10, 0.25
fracMarkers  = {'o', '+', 'x'};    % explore_frac -- scatter only

% CleanCost baseline: crimson, distinct from every COMBO colour, drawn as a reference anchor.
cleanColor = [0.70 0.15 0.20];

% TrueStar and KPAXCap sweep the low two caps only, so two entries each.
amberRamp = [0.95 0.62 0.24;     % cap 0.03 (lighter)
             0.66 0.30 0.03];    % cap 0.10 (darker)

% KPAXCap: grey-green, distinct from both the steel ramp and near-black KPAX it is compared to.
mossRamp  = [0.58 0.73 0.53;     % cap 0.03 (lighter)
             0.24 0.44 0.26];    % cap 0.10 (darker)

% --- Build the series arrays: (planner, delta) pairs, 27 in total ---
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
% Each series' goal_frontier_size, NaN for anything that is not a CountingStars arm. The budget
% figure divides budget_used by it, so it has to travel WITH the series rather than be re-derived:
% B is not in the CSV (it is a per-run setting, not a per-iteration measurement), and re-parsing it
% out of the label would put a second copy of the token convention in the plot script.
plannerGoalFrontier = [];

for di = 1:numel(deltas)
    dWidth = deltaWidths(di);
    dTag   = deltaTitles{di};
    dOne   = deltaSingleCap(di);   % this delta ran --single-point: only capDerived exists
    dPlus  = deltaPlusOnly(di);    % this delta ran --only-kinopaxplus: no other arm exists here

    if ~dPlus

    % --- CountingStars: goal_frontier_size x explore_frac, a full factorial ---
    for bi = 1:numel(csGoalFrontierSizes)
        for ei = 1:numel(csExploreFracs)
            goalF = csGoalFrontierSizes(bi);
            eFrac = csExploreFracs(ei);

            % Mirror countingStarsSkip(): FULL FACTORIAL, so --single-point is the only skip.
            if dOne && ~(goalF == csDerivedGoalFrontier && eFrac == csDerivedExploreFrac)
                continue;
            end

            plannerNames{end + 1}   = sprintf('CountingStars_B%d_e%d', goalF, eFrac); %#ok<SAGROW>
            plannerDisplay{end + 1} = sprintf('CS B%d frac%g [%s]', ...
                                              goalF, eFrac / 100, dTag); %#ok<SAGROW>
            plannerColors(end + 1, :) = budgetColors(bi, :);   %#ok<SAGROW>
            plannerStyles{end + 1}    = fracStyles{ei};        %#ok<SAGROW>
            plannerMarkers{end + 1}   = fracMarkers{ei};       %#ok<SAGROW>
            % The smallest-budget arm is drawn thicker: it is the reference the rest is read
            % against, and the one closest to KinoPaxPlus's regime.
            if goalF == min(csGoalFrontierSizes)
                plannerWidths(end + 1) = dWidth + 0.8;         %#ok<SAGROW>
            else
                plannerWidths(end + 1) = dWidth;               %#ok<SAGROW>
            end
            plannerBaseline(end + 1) = false;                  %#ok<SAGROW>
            plannerDeltaIdx(end + 1) = di;                     %#ok<SAGROW>
            plannerGoalFrontier(end + 1) = goalF;              %#ok<SAGROW>
        end
    end

    % --- CleanCost baseline: ONE point, the reference the COMBO grid is read against ---
    plannerNames{end + 1}   = sprintf('KinoPaxSTARCleanCost_r2%s_w%d_k%d_cap%d', ...
                                      cleanBaseR2, cleanBaseW, cleanBaseK, cleanBaseCap); %#ok<SAGROW>
    plannerDisplay{end + 1} = sprintf('CleanCost w%g k%g cap%g [%s]', cleanBaseW / 100, ...
                                      cleanBaseK / 100, cleanBaseCap / 100, dTag); %#ok<SAGROW>
    plannerColors(end + 1, :) = cleanColor;      %#ok<SAGROW>
    plannerStyles{end + 1}    = '-';             %#ok<SAGROW>
    plannerMarkers{end + 1}   = 'p';             %#ok<SAGROW>
    plannerWidths(end + 1)    = dWidth + 0.6;    %#ok<SAGROW>
    plannerBaseline(end + 1)  = true;            %#ok<SAGROW>
    plannerDeltaIdx(end + 1)  = di;              %#ok<SAGROW>
    plannerGoalFrontier(end + 1) = NaN;          %#ok<SAGROW>

    % --- TrueStar ---
    % --- KPAXCap ---
    for ci = 1:numel(kpaxCapCaps)
        if dOne && kpaxCapCaps(ci) ~= capDerived, continue; end
        plannerNames{end + 1}   = sprintf('KPAXCap_cap%d', kpaxCapCaps(ci)); %#ok<SAGROW>
        plannerDisplay{end + 1} = sprintf('KPAXCap cap%g [%s]', kpaxCapCaps(ci) / 100, dTag); %#ok<SAGROW>
        plannerColors(end + 1, :) = mossRamp(ci, :);    %#ok<SAGROW>
        plannerStyles{end + 1}    = '-';                %#ok<SAGROW>
        plannerMarkers{end + 1}   = 'v';                %#ok<SAGROW>
        plannerWidths(end + 1)    = dWidth + 0.6;       %#ok<SAGROW>
        plannerBaseline(end + 1)  = true;               %#ok<SAGROW>
        plannerDeltaIdx(end + 1)  = di;                 %#ok<SAGROW>
        plannerGoalFrontier(end + 1) = NaN;             %#ok<SAGROW>
    end

    % --- KPAX baseline. Gated with the rest: a --only-kinopaxplus delta does not run it. ---
    plannerNames    = [plannerNames,   {'KPAX'}];                                         %#ok<AGROW>
    plannerDisplay  = [plannerDisplay, {sprintf('KPAX [%s]', dTag)}];                     %#ok<AGROW>
    plannerColors   = [plannerColors;  0.10 0.10 0.10];                                   %#ok<AGROW>
    plannerStyles   = [plannerStyles,  {'-'}];                                            %#ok<AGROW>
    plannerMarkers  = [plannerMarkers, {'s'}];                                            %#ok<AGROW>
    plannerWidths   = [plannerWidths,  dWidth + 1.1];                                     %#ok<AGROW>
    plannerBaseline = [plannerBaseline, true];                                            %#ok<AGROW>
    plannerDeltaIdx = [plannerDeltaIdx, di];                                              %#ok<AGROW>
    plannerGoalFrontier = [plannerGoalFrontier, NaN];                                     %#ok<AGROW>

    end   % ~dPlus

    % --- KinoPaxPlus. THE ONLY ARM THAT RUNS AT EVERY DELTA, which is the entire reason the two
    % finer deltas exist in this sweep: its advantage is a tiny frontier at a fine discretisation,
    % and that is exactly what CountingStars' react_count is trying to reproduce. ---
    plannerNames    = [plannerNames,   {'KinoPaxPlus'}];                                  %#ok<AGROW>
    plannerDisplay  = [plannerDisplay, {sprintf('KinoPaxPlus [%s]', dTag)}];              %#ok<AGROW>
    plannerColors   = [plannerColors;  0.20 0.40 0.80];                                   %#ok<AGROW>
    plannerStyles   = [plannerStyles,  {'--'}];                                           %#ok<AGROW>
    plannerMarkers  = [plannerMarkers, {'d'}];                                            %#ok<AGROW>
    plannerWidths   = [plannerWidths,  dWidth + 1.1];                                     %#ok<AGROW>
    plannerBaseline = [plannerBaseline, true];                                            %#ok<AGROW>
    plannerDeltaIdx = [plannerDeltaIdx, di];                                              %#ok<AGROW>
    plannerGoalFrontier = [plannerGoalFrontier, NaN];                                     %#ok<AGROW>
end

numRunsPer = 50 * ones(1, numel(plannerNames));   % max runs searched (missing files skipped)

MAX_FLOAT_THRESH = 1e30;   % best_cost sentinel (MAX_FLOAT / INFINITY) -> NaN
numTimeSamples   = 500;

% Reference line for the growth-schedule panel. MUST match the MAX_TREE_SIZE / MAX_ITER that
% run_combo_tuning_sweep.sh writes into config.h -- neither is in the CSV, so this is the one place
% the plot has to be told. Only used to draw the dashed ideal; nothing else depends on it.
maxTreeSize = 3000000;
growthIters = 300;

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

        %% ---------- FIGURE: normalization diagnostics ----------
        % The direct evidence for both fixes. score_floor should sit flat at EPSILON = 0.01 for KPAX
        % and decay as 1/N_active for KPAXCap / TrueStar / CleanCost; cost_scale is CleanCost's
        % global denominator, whose size relative to the per-region spreads tells you which k range
        % to sweep next. Series lacking either column are skipped by getCol/plotMeanTime.
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Normalization Diagnostics (%s)', envTitle, costTitle), ...
               'Position', [90 90 1400 620]);
        subplot(1, 2, 1); hold on;
        if tmax > 0
            for pi = 1:nPlanner
                plotMeanTime(R{pi}, 'score_floor', ct, plannerColors(pi, :), ...
                             plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
            end
        end
        set(gca, 'YScale', 'log');
        xlabel('Elapsed Time (ms)'); ylabel('Syclop score floor'); grid on;
        title('score\_floor: flat 0.01 = legacy EPSILON, decaying = 1/N\_active');

        subplot(1, 2, 2); hold on;
        if tmax > 0
            for pi = 1:nPlanner
                plotMeanTime(R{pi}, 'cost_scale', ct, plannerColors(pi, :), ...
                             plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
            end
        end
        xlabel('Elapsed Time (ms)'); ylabel('D\_global (global mean cost - global min cost)'); grid on;
        title('cost\_scale: CleanCost''s costProbExpGlobal denominator, CountingStars'' distance denominator');
        clickableLegend();

        %% ---------- FIGURE: IS THE BUDGET MET ----------
        % THE CLAIM THE WHOLE DESIGN RESTS ON, and therefore the first figure to read. B is an
        % INPUT, not a cap: the doors fill it in priority order and F is supposed to come out at B
        % by construction. Everything else on this grid is a tuning question; this one is a
        % correctness question.
        %
        %   budget_used / B == 1   the budget is met. This is the expected picture at B = 50000.
        %   budget_used / B  < 1   SHORTFALL -- a door is not filling its share. Read the door panel
        %                          on the right to see which one ran dry.
        %   budget_used / B  > 1   OVERSHOOT -- the two uncapped doors (optimal, guarantee) already
        %                          exceeded B on their own. EXPECTED wherever B <= NUM_R1_REGIONS,
        %                          because both are bounded by the region count and not by B: one
        %                          node per region can be a region best, one per uncovered region
        %                          can be guaranteed. B binds only above NUM_R1_REGIONS, which is
        %                          exactly why the low-B points are on the grid -- the gap measures
        %                          how much of the frontier the priority doors take before the draw
        %                          is offered anything.
        %
        % The ratio, not the raw count, because B differs by series -- one yline at 1 then reads for
        % every arm at once.
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Budget vs Doors (%s)', envTitle, costTitle), ...
               'Position', [120 120 1400 620]);
        subplot(1, 2, 1); hold on;
        for pi = 1:nPlanner
            B = plannerGoalFrontier(pi);
            if isnan(B), continue; end   % not a CountingStars series: no budget to check
            plotMeanIter(R{pi}, @(t) getCol(t, 'budget_used') / B, ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        set(gca, 'YScale', 'log'); grid on;
        yline(1, 'k--', 'budget met', 'LineWidth', 1.4, 'HandleVisibility', 'off');
        xlabel('Iteration'); ylabel('budget\_used / goal\_frontier\_size');
        title({'Is the budget met?', 'below 1 = a door ran dry; above 1 = the optimal door alone overran B'});
        clickableLegend();

        % --- Which door filled it. Every node came through a named door and the counts are exact,
        % so a shortfall on the left has an address here.
        %
        %   optimal_count      the top door, uncapped, first claim every iteration.
        %   admitted_explore   the freshness door, spending explore_frac of what optimal left.
        %   reactivated_best   the guarantee, realised. Compare with guaranteed_react (the PLANNED
        %                      count that set the draw's probability): the gap is the guaranteed
        %                      nodes Part B skipped for already being in the frontier.
        %   reactivated_count  the uniform draw, filling the remainder.
        subplot(1, 2, 2); hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) getCol(t, 'optimal_count'), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
            plotMeanIter(R{pi}, @(t) getCol(t, 'admitted_explore'), ...
                         plannerColors(pi, :), '--', max(0.5, plannerWidths(pi) - 0.6), '');
            plotMeanIter(R{pi}, @(t) getCol(t, 'reactivated_best'), ...
                         plannerColors(pi, :), '-.', max(0.5, plannerWidths(pi) - 0.3), '');
            plotMeanIter(R{pi}, @(t) getCol(t, 'reactivated_count'), ...
                         plannerColors(pi, :), ':', max(0.5, plannerWidths(pi) - 0.6), '');
        end
        set(gca, 'YScale', 'log'); grid on;
        xlabel('Iteration'); ylabel('nodes');
        title({'optimal (solid), explore (dashed), guarantee (dash-dot), draw (dotted)', ...
               'the four doors, in the order they are offered the budget'});

        %% ---------- FIGURE: is freshness still scarce ----------
        % ord_cutoff is the freshness threshold the remaining budget bought this iteration: a
        % candidate is admitted by the freshness door when its REGION's population is below it.
        %
        %   RISING over a run   expected. Regions fill, so buying the same number of nodes costs a
        %                       looser threshold every iteration.
        %   PINNED AT 0         no non-optimal candidate is ever fresh enough, and explore_frac is
        %                       doing nothing. Either the optimal door is taking the whole budget
        %                       (check optimal_count against B on the previous figure) or every
        %                       region is already populated.
        %   AT 256              saturated: the whole candidate pool is fresher than X demands, so
        %                       every non-optimal candidate is admitted. explore_frac is not binding.
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Freshness Cutoff and Cost Scale (%s)', envTitle, costTitle), ...
               'Position', [140 140 1400 620]);
        subplot(1, 2, 1); hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) getCol(t, 'ord_cutoff'), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        grid on;
        yline(256, 'k--', 'saturated (all fresh)', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        xlabel('Iteration'); ylabel('ord\_cutoff (region population)');
        title({'Freshness cutoff', 'rising = regions filling; 0 = explore\_frac inert; 256 = not binding'});
        clickableLegend();

        % --- The denominator of the top door's test. distance = (cost - regionMin) / cost_scale,
        % and distance 0 is what the optimal door admits -- so a cost_scale collapsing toward 0 is
        % the one way that test could go degenerate without anything else looking wrong.
        subplot(1, 2, 2); hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) getCol(t, 'cost_scale'), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        grid on;
        xlabel('Iteration'); ylabel('D\_global (global mean cost - global min cost)');
        title({'cost\_scale: the distance denominator', 'collapsing toward 0 would make distance 0 degenerate'});

        %% ---------- FIGURE: budget invariants ----------
        % Three checks that decide whether a tuning conclusion from this run is trustworthy at all.
        %
        % All three are per-ITERATION quantities, so they use plotMeanIter -- the mean across runs
        % at each iteration index -- and draw ONE LINE PER SERIES. The first version drew one raw
        % line per RUN with no legend, i.e. ~60 unlabelled lines, which is why this figure was
        % unreadable.
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Budget Invariants (%s)', envTitle, costTitle), ...
               'Position', [150 150 1560 640]);

        % --- 1. Is propagate still on kernel1? ---
        % prop_attempted / frontier_repeat_size is exactly 32 on the kernel1 path (one 32-thread
        % block per repeat entry) and h_propIterations_ < 32 on kernel2. h_propIterations_ ALONE is
        % not a valid detector: it is only assigned inside the kernel2 branch, so on the kernel1
        % path it still holds a stale value from whichever earlier iteration last took kernel2.
        subplot(1, 3, 1); hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) safeRatio(getCol(t, 'prop_attempted'), ...
                                               getCol(t, 'frontier_repeat_size')), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        yline(32, 'k--', 'kernel1', 'LineWidth', 1.2);
        ylim([0 36]);
        xlabel('Iteration'); ylabel('propagations per repeat entry'); grid on;
        title({'Kernel1 check: 32 while the ceiling holds', 'below 32 = kernel2 (block split across candidates)'});

        % --- 2. Is the controller tracking its growth schedule? ---
        subplot(1, 3, 2); hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) getCol(t, 'tree_size'), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        plot([0 growthIters], [0 maxTreeSize], 'k--', 'LineWidth', 1.2, ...
             'DisplayName', 'linear schedule');
        xlabel('Iteration'); ylabel('tree\_size'); grid on;
        title({'Tree growth vs the linear schedule', 'shortfall = demand exceeds the candidate pool'});

        % --- 3. What is the frontier actually made of? ---
        % `reactivated` counts frontier bits among the PRE-EXISTING tree, i.e. exactly Part B's
        % output, so this is the share of the frontier that is re-expansion rather than new nodes.
        %
        % READ THIS PANEL FIRST WHEN KERNEL1 FAILS EARLY. Part B's guarantee is unconditional for
        % an UNCOVERED region, so F has a floor at the number of regions the optimal door missed.
        % Since rep >= 1, frontierRepeatSize >= F, and kernel2 is forced once 32*F > remaining
        % whatever the budget says. A curve pinned near 100% means F is reactivation-dominated: the
        % admissions are a rounding error next to the guarantee and the draw, and B is being met by
        % re-expansion rather than by new ground.
        subplot(1, 3, 3); hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) 100 * safeRatio(getCol(t, 'reactivated'), ...
                                                     getCol(t, 'frontier_size')), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        ylim([0 105]);
        xlabel('Iteration'); ylabel('% of frontier from Part B'); grid on;
        title({'Frontier composition: reactivated / frontier\_size', 'near 100% = the region-best guarantee dominates F'});
        clickableLegend();

        %% ---------- FIGURE: is the frontier small ----------
        % THE SECOND QUESTION, AND THE ONE THE DESIGN TURNS ON. KinoPaxPlus wins by dividing the
        % whole propagation budget over a frontier its pruning keeps tiny: bf = MAX_TREE_SIZE/(F*32)
        % reaches 40,000 propagations per node at F = 10. A frontier pinned near nActive gets ~32.
        % No fan-out weighting closes three orders of magnitude -- only a smaller F does.
        %
        % prop_per_node = prop_attempted / frontier_size IS THE DIRECT COMPARISON, AND THE POINT OF
        % CONTROLLING F IS CONTROLLING IT. If it does not move across the 25x span in B, B is not
        % the lever and no other knob on this grid matters.
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Frontier Size and Focus (%s)', envTitle, costTitle), ...
               'Position', [180 180 1500 640]);

        subplot(1, 2, 1); hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) getCol(t, 'frontier_size'), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        set(gca, 'YScale', 'log'); grid on;
        xlabel('Iteration'); ylabel('frontier size F');
        title({'Frontier size', 'goal\_frontier\_size is the input; everything else follows from F'});

        subplot(1, 2, 2); hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) safeRatio(getCol(t, 'prop_attempted'), ...
                                               getCol(t, 'frontier_size')), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        set(gca, 'YScale', 'log'); grid on;
        yline(32, 'k--', 'one block', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        xlabel('Iteration'); ylabel('propagations per frontier node');
        title({'Focus: prop\_attempted / frontier\_size', 'must move with B; compare against KinoPaxPlus bf'});
        clickableLegend();

        %% ---------- FIGURE: fan-out budget ----------
        % block_scale is the fraction of each node's requested BOOST that survived the buffer
        % ceiling; the rep >= 1 floor is never scaled, so no frontier node is ever left blockless.
        %
        %   block_scale = 1     the buffer ceiling never bound; the design budget alone is setting
        %                       fan-out (blockBudget = maxBlocks * B, split at admission).
        %   block_scale < 1     the BUFFER is setting it, not the rule.
        %   block_scale near 0  the rep >= 1 floor ate the budget and the fan-out split is inert.
        %                       That is a goal_frontier_size problem, and explore_frac will not
        %                       move it.
        %
        % block_ceiling below frontier_size is the same story stated in absolute terms.
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Fan-out Budget (%s)', envTitle, costTitle), ...
               'Position', [200 200 1500 640]);

        subplot(1, 2, 1); hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) getCol(t, 'block_scale'), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        ylim([0 1.05]); grid on;
        yline(1, 'k--', 'ceiling not binding', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        xlabel('Iteration'); ylabel('block\_scale');
        title({'Did the buffer cap the boost?', 'near 0 = the frontier ate the budget (a B problem)'});

        subplot(1, 2, 2); hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) getCol(t, 'block_ceiling'), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
            plotMeanIter(R{pi}, @(t) getCol(t, 'frontier_size'), ...
                         plannerColors(pi, :), ':', max(0.5, plannerWidths(pi) - 0.6), '');
        end
        set(gca, 'YScale', 'log'); grid on;
        xlabel('Iteration'); ylabel('blocks');
        title({'block\_ceiling (solid) vs frontier\_size (dotted)', 'ceiling below frontier = nothing left to concentrate'});
        clickableLegend();

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
                % CountingStars first: its labels are CountingStars_B<budget>_e<exploreFrac>.
                % The list is a WHITELIST on purpose -- an unrecognised label is a typo or a grid
                % that drifted, and erroring here is far better than silently loading nothing and
                % reporting "0 runs" for a series that was actually written.
                if startsWith(planner, 'CountingStars') || startsWith(planner, 'KinoPaxSTAR') ...
                        || startsWith(planner, 'KPAXCap')
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

function plotMeanIter(runs, valueFcn, color, style, width, name)
    % Mean of a per-ITERATION quantity across a series' runs, drawn as ONE line.
    %
    % The iteration-domain counterpart of plotMeanTime. Used for the budget invariants, which are
    % properties of an iteration (a launch configuration, a budget) and not of elapsed time, so
    % resampling them onto a time grid would blur exactly the step changes worth seeing.
    %
    % valueFcn(tbl) returns a per-iteration column, or [] when the run lacks the columns it needs --
    % baselines write NaN for every COMBO-only column, so those series simply do not draw rather
    % than erroring.
    %
    % Runs are RAGGED: the 6 s timeout ends them at different iterations. Each iteration index is
    % averaged over whatever runs reached it, then the tail is trimmed where fewer than half the
    % runs did, which stops the right edge silently degenerating into one long run's trace -- the
    % same failure plotMeanTime's right-tail hold exists to prevent.
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

function r = safeRatio(a, b)
    % Elementwise a./b guarding both a missing column and a zero denominator. Returns [] when
    % either column is absent, which plotMeanIter treats as "this series has nothing to draw".
    if isempty(a) || isempty(b), r = []; return; end
    n = min(numel(a), numel(b));
    a = a(1:n); b = b(1:n);
    r = NaN(n, 1);
    ok = b > 0;
    r(ok) = a(ok) ./ b(ok);
end
