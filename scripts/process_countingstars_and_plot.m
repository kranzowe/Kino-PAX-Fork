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

% COMBO grid - must match ACC_GAINS / FAN_GAINS / REACT_FRAC in
% kinopaxstar_combo_tuning_sweep.cu. Values are the integer label tokens (100 x the float), exactly
% as they appear in the filenames. cross_check_combo_grid.py asserts these stay in step; when they
% drift, MATLAB reports "0 runs" for the orphaned series rather than erroring, which is the failure
% mode that silently wastes a whole sweep.
%
% COMBO runs TWO shapes: acceptance (which nodes join) and fan-out (where propagation goes).
%
% react_count -- the GLOBAL FRONTIER CAP -- is the headline axis, because it sets F and F sets
% propagations-per-node. KinoPaxPlus divides the whole budget over a frontier its pruning keeps
% tiny (bf = MAX_TREE_SIZE/(F*32), 40,000 propagations per node at F = 10); COMBO's frontier was
% pinned at F >= nActive and got ~32. No fan-out weighting closes a gap like that -- only a smaller
% F does. react 0 means the frontier is exactly this iteration's admissions.
%
% halfLife is the fan-out decay for the explore door: blocks = max(maxBlocks >> (ordinal/halfLife), 1),
% so 1 gives 15, 7, 3, 1 ... and 2 stretches it. explore_count is the novelty quota per region.
%
% The grid is a CROSS, not a full factorial: react x halfLife at the derived explore, plus explore
% swept at the derived react/halfLife. Values are the label tokens exactly as they appear in the
% filenames -- react and halfLife plain, explore as round(100 x float).
% cross_check_countingstars_grid.py asserts these stay in step with the .cu and the .sh; when they
% drift, MATLAB reports "0 runs" for the orphaned series rather than erroring, which is the failure
% mode that silently wastes a whole sweep.
csReactCounts   = [1000 100000];
csHalfLives     = [1 4];
csExploreCounts = [100 1000];
csMaxBlocks     = [12 32];

% The derived operating point that --single-point selects.
csDerivedReact     = 1000;
csDerivedHalfLife  = 1;
csDerivedExplore   = 100;
csDerivedMaxBlocks = 12;

% CleanCost baseline point - one series, the well-tuned operating point. Same label format as the
% cost sweep, so its historical CSVs load here unchanged.
cleanBaseR2  = 'off';
cleanBaseW   = 90;
cleanBaseK   = 100;
cleanBaseCap = 3;

% TrueStar and KPAXCap cap sweeps - must match TRUE_CAPS / KPAXCAP_CAPS in the benchmark.
kpaxCapCaps = [3];

% colour = react_count (frontier size, the headline axis), line style = halfLife, marker =
% explore_count. react gets the channel the eye reads first because F is what the whole design turns
% on: near-black is react 0, the arm where the frontier is only this iteration's admissions.
% FOUR axes, three style channels, so colour carries the pair that actually interacts: react
% (frontier size) x maxBlocks (ramp height). Darker = smaller frontier, which is the direction the
% whole design is pushing.
%   rows: (react 1e3, b12) (react 1e3, b32) (react 1e5, b12) (react 1e5, b32)
reactBlockColors = [0.10 0.10 0.10;    % react 1e3, maxBlocks 12  (smallest frontier, shortest ramp)
                    0.16 0.38 0.63;    % react 1e3, maxBlocks 32
                    0.55 0.42 0.20;    % react 1e5, maxBlocks 12
                    0.75 0.72 0.80];   % react 1e5, maxBlocks 32  (largest frontier, tallest ramp)
halfStyles = {'-', '--'};         % halfLife = 1 (sharp), 4 (stretched)
exploreMarkers = {'o', '+'};          % explore 1, 10 -- scatter only

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

for di = 1:numel(deltas)
    dWidth = deltaWidths(di);
    dTag   = deltaTitles{di};
    dOne   = deltaSingleCap(di);   % this delta ran --single-point: only capDerived exists
    dPlus  = deltaPlusOnly(di);    % this delta ran --only-kinopaxplus: no other arm exists here

    if ~dPlus

    % --- CountingStars: react x halfLife x explore, as a cross ---
    for bi = 1:numel(csMaxBlocks)
        for ri = 1:numel(csReactCounts)
            for hi = 1:numel(csHalfLives)
                for ei = 1:numel(csExploreCounts)
                    maxB    = csMaxBlocks(bi);
                    react   = csReactCounts(ri);
                    half    = csHalfLives(hi);
                    explore = csExploreCounts(ei);

                    % Mirror countingStarsSkip(): FULL FACTORIAL, so --single-point is the only skip.
                    if dOne && ~(react == csDerivedReact && half == csDerivedHalfLife && ...
                                 explore == csDerivedExplore && maxB == csDerivedMaxBlocks)
                        continue;
                    end

                    plannerNames{end + 1}   = sprintf('CountingStars_b%d_r%d_h%d_e%d', ...
                                                      maxB, react, half, explore); %#ok<SAGROW>
                    plannerDisplay{end + 1} = sprintf('CS b%d react%g h%d e%g [%s]', ...
                                                      maxB, react, half, explore / 100, dTag); %#ok<SAGROW>
                    % Colour index over the (react, maxBlocks) pair -- see reactBlockColors.
                    plannerColors(end + 1, :) = reactBlockColors((ri - 1) * numel(csMaxBlocks) + bi, :); %#ok<SAGROW>
                    plannerStyles{end + 1}    = halfStyles{hi};        %#ok<SAGROW>
                    plannerMarkers{end + 1}   = exploreMarkers{ei};    %#ok<SAGROW>
                    % The smallest-frontier arm is drawn thicker: it is the reference the rest is
                    % read against, and the one closest to KinoPaxPlus's regime.
                    if react == min(csReactCounts)
                        plannerWidths(end + 1) = dWidth + 0.8;         %#ok<SAGROW>
                    else
                        plannerWidths(end + 1) = dWidth;               %#ok<SAGROW>
                    end
                    plannerBaseline(end + 1) = false;                  %#ok<SAGROW>
                    plannerDeltaIdx(end + 1) = di;                     %#ok<SAGROW>
                end
            end
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
        title('cost\_scale: the costProbExpGlobal denominator (CleanCost only)');
        clickableLegend();

        %% ---------- FIGURE: which door built the tree ----------
        % THE FIRST QUESTION THIS PLANNER EXISTS TO ANSWER. COMBO admitted with one global
        % probability and could not say why a node got in; here every node came through a named
        % door and the counts are exact.
        %
        %   admitted_explore  novelty. Collapsing toward zero early means the R2 cells are
        %                     saturating and explore_count is not the binding constraint -- the
        %                     discretisation is. That is a delta question, not a tuning one.
        %   admitted_cost     region bests. Pinned near nActive is expected and healthy.
        %   reactivated_count Part B. Should track react_count in expectation; a persistent gap
        %                     means the uniform draw is not seeing the population it should.
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Admission Doors (%s)', envTitle, costTitle), ...
               'Position', [120 120 1400 620]);
        subplot(1, 2, 1); hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) getCol(t, 'admitted_explore'), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        set(gca, 'YScale', 'log'); grid on;
        xlabel('Iteration'); ylabel('nodes admitted by the explore door');
        title({'Novelty admissions', 'falling to zero = R2 cells saturated, not a tuning limit'});
        clickableLegend();

        subplot(1, 2, 2); hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) getCol(t, 'admitted_cost'), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
            plotMeanIter(R{pi}, @(t) getCol(t, 'reactivated_count'), ...
                         plannerColors(pi, :), ':', max(0.5, plannerWidths(pi) - 0.6), '');
            plotMeanIter(R{pi}, @(t) getCol(t, 'reactivated_best'), ...
                         plannerColors(pi, :), '-.', max(0.5, plannerWidths(pi) - 0.3), '');
        end
        set(gca, 'YScale', 'log'); grid on;
        xlabel('Iteration'); ylabel('nodes');
        title({'admitted\_cost (solid), reactivated\_count (dotted), reactivated\_best (dash-dot)', ...
               'best tracking frontier\_size = the guarantee IS the frontier'});

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
        % READ THIS PANEL FIRST WHEN KERNEL1 FAILS EARLY. Part B's region-best branch is
        % UNCONDITIONAL -- one node per explored region, up to NUM_R1_REGIONS, outside h_reactFrac_
        % entirely -- so F has a floor at nActive. Since rep >= 1, frontierRepeatSize >= F, and
        % kernel2 is forced once 32*F > remaining no matter what repTarget does. A curve pinned near
        % 100% here means F is region-best dominated and no acceptance tuning will move the kernel1
        % crossover; the levers are the guarantee itself or the R1 grid size.
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
        % prop_per_node = prop_attempted / frontier_size IS THE DIRECT COMPARISON. If it stays in
        % the tens across every react setting, react_count is not doing its job and no other knob
        % on this grid matters.
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
        title({'Frontier size', 'react\_count is the knob; everything else follows from F'});

        subplot(1, 2, 2); hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) safeRatio(getCol(t, 'prop_attempted'), ...
                                               getCol(t, 'frontier_size')), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        set(gca, 'YScale', 'log'); grid on;
        yline(32, 'k--', 'one block', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        xlabel('Iteration'); ylabel('propagations per frontier node');
        title({'Focus: prop\_attempted / frontier\_size', 'compare against KinoPaxPlus bf; tens = not focused'});
        clickableLegend();

        %% ---------- FIGURE: fan-out budget ----------
        % block_scale is the fraction of each node's requested BOOST that survived the buffer
        % ceiling; the rep >= 1 floor is never scaled, so no frontier node is ever left blockless.
        %
        %   block_scale = 1     the ceiling never bound; halfLife alone is setting fan-out.
        %   block_scale < 1     the BUFFER is setting it, not the rule.
        %   block_scale near 0  F itself has eaten the budget. That is a react_count problem, and
        %                       no halfLife or explore_count setting will move it.
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
        title({'Did the buffer cap the boost?', 'near 0 = the frontier ate the budget (a react\_count problem)'});

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
                % CountingStars first: its labels are CountingStars_r<react>_h<half>_e<explore>.
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
