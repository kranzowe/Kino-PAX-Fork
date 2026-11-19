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
%   cd build/Data/Benchmarks/KinoPaxStarComboTuning/zigzag     % or .../house
%   addpath('<repo>/scripts')
%   process_combo_tuning_and_plot
% run('<abs path>/process_combo_tuning_and_plot.m') would cd to the scripts folder
% first, and dataDir below ('' = current folder) would then find nothing.

clear; clc; close all;

%% --- Configuration ---
dataDir = '';   % '' = current directory (run this from Data/Benchmarks/KinoPaxStarComboTuning)

% One environment per run — must match the subfolder you cd'd into.
%   'zigzag' -> 'Zigzag Corridor',  'house' -> 'House'
% SCOPE: zigzag and narrowPassage this pass (matches ENV_NAMES in run_combo_tuning_sweep.sh).
% ONE PER RUN -- each environment writes to its own subfolder, so set this to match the folder you
% cd'd into and re-run for the other.
%   'zigzag' -> 'Zigzag Corridor',  'narrowPassage' -> 'Narrow Passage',  'house' -> 'House'
environments = {'zigzag'};
envTitles    = {'Zigzag Corridor'};
% environments = {'narrowPassage'};   envTitles = {'Narrow Passage'};

% Cost metric axis — one build each, so one set of figures each.
metrics      = {'effort', 'length'};
metricTitles = {'Control Effort', 'Workspace Path Length'};
metricYLabels = {'Path Cost (control effort)', 'Path Cost (workspace path length)'};

% Delta axis — OVERLAID inside each figure, encoded as line style. The filename token is
% sprintf('%s_%s', delta, metric), e.g. 'fine_control_length'.
% SCOPE: coarse delta only this pass. Full set preserved on the commented line below -- swap the
% two blocks (here and in deltaSingleCap) to restore, exactly as in run_combo_tuning_sweep.sh.
deltas      = {'large'};
deltaTitles = {'27k'};
deltaWidths = [1.4];      % one per delta; line WIDTH is the delta channel now that k took style
% deltas      = {'large', 'fine', 'fine_control'};
% deltaTitles = {'27k', '216k W-refined', '216k V-refined'};
% deltaWidths = [1.0, 1.8, 2.6];
% Which caps exist at each delta: the coarse delta sweeps the axis, the finer ones ran --single-point
% so only the derived point exists. Must match DELTA_EXTRA_ARGS in run_combo_tuning_sweep.sh.
capDerived     = 10;      % label token for cap = 0.1 (CAP_DERIVED in the benchmark)
% true where run_combo_tuning_sweep.sh passes --single-point for that delta, i.e. only capDerived was
% run there. Mirrors capSkip() in the benchmark; each planner then filters its OWN cap list, which
% matters now that CleanCost's caps and TrueStar/KPAXCap's caps are different sets.
deltaSingleCap = [false];
% deltaSingleCap = [false, true, true];   % restore alongside the full delta set

deltaLabel = '3 deltas overlaid';

% COMBO grid - must match ACC_GAINS / FAN_GAINS / REACT_FRAC in
% kinopaxstar_combo_tuning_sweep.cu. Values are the integer label tokens (100 x the float), exactly
% as they appear in the filenames. cross_check_combo_grid.py asserts these stay in step; when they
% drift, MATLAB reports "0 runs" for the orphaned series rather than erroring, which is the failure
% mode that silently wastes a whole sweep.
%
% COMBO runs TWO shapes: acceptance (which nodes join) and fan-out (where propagation goes).
%
% N -- the SIGMA MULTIPLE -- is the headline axis. A node is favoured, and gets h_repeatMax_ blocks
% instead of 1, when its fan-out score exceeds mu + N*sigma over the score distribution of the whole
% realised frontier. So N sets how far into the tail the boost reaches, in units of the
% distribution's own spread, and kFan decides how much spread there is to reach into.
%
% WHY SIGMA. Two earlier rules put the step in the wrong place. Thresholding at the MEAN of each
% delta favoured the MAJORITY -- both deltas are right-skewed, so mean > median and the measured
% favoured fraction came out above 0.5 at every gain. Feeding a threshold back toward a target
% fraction phi fixed the placement but sized the block budget against a favoured count measured on
% the wrong population. mu + N*sigma is scale-free: a fixed N holds its place in the tail whatever
% the gains do to the spread, with nothing tracked across iterations.
%
% THE TWO AXES FIGHT AT THE TOP. Raising kFan drives the shape bimodal with mass p at the top, so
% mu -> p, sigma -> sqrt(p(1-p)), and the largest N that still favours anyone is sqrt((1-p)/p). Past
% it n_fav is 0 and the run is flat-1-block, which reads as "the boost did nothing". The fan_n_max
% column logs that ceiling per iteration, so plot it whenever a high-N/high-kFan point looks inert.
%
% kFan = 0 (token 0) is the UNIFORM control arm: every score is identical, sigma is 0, and the
% planner's degenerate branch gives every frontier node the same block count.
comboFanSigmaN = [200 300 400 500];    % both axes moved up: the previous grid won at both edges
comboFanGains  = [0 1600 3200 6400];   % 0 = uniform control arm
comboAccGain   = 400;                  % fixed this pass
comboReact     = 10;

% The derived operating point that --single-point selects (N = 1.5, kFan = 4).
comboDerivedSigmaN = 200;
comboDerivedFan    = 1600;

% CleanCost baseline point - one series, the well-tuned operating point. Same label format as the
% cost sweep, so its historical CSVs load here unchanged.
cleanBaseR2  = 'off';
cleanBaseW   = 90;
cleanBaseK   = 100;
cleanBaseCap = 3;

% TrueStar and KPAXCap cap sweeps - must match TRUE_CAPS / KPAXCAP_CAPS in the benchmark.
trueCaps    = [3 10];
kpaxCapCaps = [3 10];

% colour = N (how far into the tail), line style = kFan (how much spread), width = delta base.
% N gets the channel the eye reads first because it is the headline axis: the question is whether
% concentrating propagation on a few nodes beats spreading it.
%   N 0     - near-black: the threshold-at-the-mean arm, which favours the majority
%   rising  - steel ramp darkening as the favoured set shrinks
sigmaNColors = [0.62 0.76 0.90;    % N 2.0   (the derived operating point, previous best)
                0.40 0.60 0.82;    % N 3.0
                0.24 0.45 0.70;    % N 4.0
                0.03 0.15 0.31];   % N 5.0   (most extreme sparsity)
fanStyles  = {'-.', ':', '-', '--'};  % kFan = 0 (uniform control), 16, 32, 64
fanMarkers = {'s', 'o', 'x', '+'};    % scatter only

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

    % --- COMBO: sigma multiple x fan-out gain ---
    for pi2 = 1:numel(comboFanSigmaN)
        for fi = 1:numel(comboFanGains)
            sigmaN = comboFanSigmaN(pi2);
            kFan   = comboFanGains(fi);

            % Mirror comboSkip(): at kFan = 0 sigma is 0 and every N runs the identical uniform
            % control arm, so only the derived N is kept there. --single-point is the only other skip.
            if kFan == 0 && sigmaN ~= comboDerivedSigmaN, continue; end
            if dOne && ~(sigmaN == comboDerivedSigmaN && kFan == comboDerivedFan), continue; end

            plannerNames{end + 1}   = sprintf('KinoPaxSTARCOMBO_sn%d_kf%d_ka%d', ...
                                              sigmaN, kFan, comboAccGain); %#ok<SAGROW>
            if kFan == 0
                sigmaTag = 'uniform';
            else
                sigmaTag = sprintf('N%g', sigmaN / 100);
            end
            plannerDisplay{end + 1} = sprintf('COMBO %s kFan%g [%s]', ...
                                              sigmaTag, kFan / 100, dTag); %#ok<SAGROW>
            plannerColors(end + 1, :) = sigmaNColors(pi2, :);     %#ok<SAGROW>
            plannerStyles{end + 1}    = fanStyles{fi};            %#ok<SAGROW>
            plannerMarkers{end + 1}   = fanMarkers{fi};           %#ok<SAGROW>
            % The uniform arm is drawn thicker: it is the reference the rest is read against.
            if kFan == 0
                plannerWidths(end + 1) = dWidth + 0.8;            %#ok<SAGROW>
            else
                plannerWidths(end + 1) = dWidth;                  %#ok<SAGROW>
            end
            plannerBaseline(end + 1) = false;                     %#ok<SAGROW>
            plannerDeltaIdx(end + 1) = di;                        %#ok<SAGROW>
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
    for ci = 1:numel(trueCaps)
        if dOne && trueCaps(ci) ~= capDerived, continue; end
        plannerNames{end + 1}   = sprintf('KinoPaxSTARTrue_cap%d', trueCaps(ci)); %#ok<SAGROW>
        plannerDisplay{end + 1} = sprintf('True cap%g [%s]', trueCaps(ci) / 100, dTag); %#ok<SAGROW>
        plannerColors(end + 1, :) = amberRamp(ci, :);   %#ok<SAGROW>
        plannerStyles{end + 1}    = '-';                %#ok<SAGROW>
        plannerMarkers{end + 1}   = '^';                %#ok<SAGROW>
        plannerWidths(end + 1)    = dWidth;             %#ok<SAGROW>
        plannerBaseline(end + 1)  = false;              %#ok<SAGROW>
        plannerDeltaIdx(end + 1)  = di;                 %#ok<SAGROW>
    end

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

    % --- Baselines, once per delta, drawn thick so they read as anchors ---
    plannerNames    = [plannerNames,   {'KPAX', 'KinoPaxPlus'}];                          %#ok<AGROW>
    plannerDisplay  = [plannerDisplay, {sprintf('KPAX [%s]', dTag), ...
                                        sprintf('KinoPaxPlus [%s]', dTag)}];              %#ok<AGROW>
    plannerColors   = [plannerColors;  0.10 0.10 0.10;  0.20 0.40 0.80];                  %#ok<AGROW>
    plannerStyles   = [plannerStyles,  {'-', '--'}];                                      %#ok<AGROW>
    plannerMarkers  = [plannerMarkers, {'s', 'd'}];                                       %#ok<AGROW>
    plannerWidths   = [plannerWidths,  dWidth + 1.1, dWidth + 1.1];                        %#ok<AGROW>
    plannerBaseline = [plannerBaseline, true, true];                                      %#ok<AGROW>
    plannerDeltaIdx = [plannerDeltaIdx, di, di];                                          %#ok<AGROW>
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

        %% ---------- FIGURE: growth controller ----------
        % The direct readout of what replaced `cap`. Expectations, from the derivation:
        %   p_target_accept   RISES over a run (~5x) as the tree buffer fills and the fan-out is
        %                     forced down. A FLAT line means it is pinned at pMax, i.e. the demand
        %                     exceeds what the candidate pool can supply.
        %   p_target_react    an order of magnitude BELOW p_target_accept - it is divided across the
        %                     whole tree, not the candidate list. If the two converge, the
        %                     reactivation flux is about to swamp admission.
        %   want_this_iter    the growth schedule's demand for this iteration. Under growthExp = 2 it
        %                     is front-loaded, so it falls through the run.
        % Every non-COMBO series lacks these columns; getCol returns [] and plotMeanTime skips it.
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Growth Controller (%s)', envTitle, costTitle), ...
               'Position', [120 120 1400 620]);
        subplot(1, 2, 1); hold on;
        if tmax > 0
            for pi = 1:nPlanner
                plotMeanTime(R{pi}, 'p_target_accept', ct, plannerColors(pi, :), ...
                             plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
                plotMeanTime(R{pi}, 'p_target_reactivate', ct, plannerColors(pi, :), ...
                             ':', max(0.5, plannerWidths(pi) - 0.6), '');
            end
        end
        set(gca, 'YScale', 'log');
        xlabel('Elapsed Time (ms)'); ylabel('acceptance budget'); grid on;
        title('p\_target\_accept (solid) and p\_target\_reactivate (dotted)');

        subplot(1, 2, 2); hold on;
        if tmax > 0
            for pi = 1:nPlanner
                plotMeanTime(R{pi}, 'want_this_iter', ct, plannerColors(pi, :), ...
                             plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
            end
        end
        xlabel('Elapsed Time (ms)'); ylabel('mean fan-out target'); grid on;
        title('rep\_target: selectivity-driven, then clamped by the kernel1 ceiling');
        clickableLegend();

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

        %% ---------- FIGURE: shape diagnostics ----------
        % The two checks specific to the split-shape design.
        %
%   fan_frac = n_fav / frontier_size, the favoured minority, counted EXACTLY over the
        %   realised frontier rather than proxied. IT MUST SIT WELL UNDER 0.5. The first fan-out
        %   rule measured it above 0.5 at every gain, which is what proved it was boosting the
        %   majority rather than an elite. Expect ~0.10-0.20 at N = 1 and ~0.02-0.05 at N = 2, and
        %   a flat 1.0 for the kFan = 0 control arm (degenerate spread: everyone favoured).
        %
        %   blend_w_cost is the normalised COST weight of the acceptance shape. It starts at 0
        %   (pure coverage) and rises toward 1 as the tree fills. IF IT NEVER APPROACHES 1 the run
        %   ended before cost ever took over, and h_blendMid_ is the lever -- not the gains.
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Shape Diagnostics (%s)', envTitle, costTitle), ...
               'Position', [180 180 1500 640]);

        subplot(1, 2, 1); hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) getCol(t, 'fan_frac'), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        ylim([0 1]); set(gca, 'YScale', 'log'); grid on;
        xlabel('Iteration'); ylabel('measured favoured fraction');
        title({'Favoured fraction: n\_fav / frontier\_size', ...
               'must sit well under 0.5, or the boost is reaching the majority'});

        subplot(1, 2, 2); hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) getCol(t, 'blend_w_cost'), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        ylim([0 1]); grid on;
        xlabel('Iteration'); ylabel('cost weight of the acceptance shape');
        title({'Coverage -> cost handover', 'never nearing 1 = lower h\_blendMid\_, not the gains'});
        clickableLegend();

        %% ---------- FIGURE: fan-out budget ----------
        % WHETHER CONCENTRATION IS EVEN POSSIBLE, and how hard it is hitting.
        %
%   fan_sigma, the spread of the fan-out scores over the realised frontier. It is what the
        %   threshold is measured in, so COLLAPSING TOWARD 0 MEANS THE SCORES CARRY NO SIGNAL and
        %   no N can separate anything -- raise kFan. A flat 0 is the kFan = 0 control arm.
        %
        %   rep_hi, the blocks a favoured node receives. h_repeatMax_ (15) is the normal value;
        %   BELOW IT MEANS THE BLOCK CEILING, NOT N, IS SETTING FAN-OUT. Pinned at 1 means
        %   block_ceiling has fallen below frontier_size -- the rep >= 1 floor has spent the whole
        %   budget, and the constraint is the frontier (F >= nActive, because Part B reactivates
        %   every region's best unconditionally), not the fan-out rule.
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Fan-out Budget (%s)', envTitle, costTitle), ...
               'Position', [200 200 1500 640]);

        subplot(1, 2, 1); hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) getCol(t, 'fan_sigma'), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
yline(0, 'k--', 'no spread', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        grid on;
        xlabel('Iteration'); ylabel('fan\_sigma (spread of the frontier scores)');
        title({'Is there anything to separate?', 'near 0 = the scores carry no signal; raise kFan'});

        subplot(1, 2, 2); hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) getCol(t, 'rep_hi'), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        yline(1, 'k--', 'no boost', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        grid on;
        xlabel('Iteration'); ylabel('rep\_hi (blocks per favoured node)');
title({'How hard a favoured node fans out', 'below 15 = the block ceiling is setting it, not N'});
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
