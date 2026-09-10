%% CountingStars v3.4 Sweep Visualization - BUG-ISOLATION harness (fixed point, not a tuning grid)
% Reads per-iteration CSVs produced by examples/gpu/countingstars_sweep.cu
% (run via scripts/run_countingstars_sweep.sh).
%
% THIS FILE NO LONGER PLOTS A COUNTINGSTARS TUNING GRID. The bufferSlope x bufferFloor x
% explore_frac x cost_frac grid that used to be plotted here already found a good point --
% (1.2, 0.3, 0.1, 0.8) -- and paper_benchmark.cu already runs CountingStars at exactly that point.
% This script's current job is reproducing paper_benchmark.cu's own comparison, at discretizations
% already confirmed not to hang/crash, to help isolate a still-open cudaErrorIllegalAddress/hang
% bug paper_benchmark.cu hit at its own `tiny` delta (see countingstars_sweep.cu's header for the
% fuller story). KinoPaxSTARTrue is the newest addition, testing whether IT -- not KPAX -- is
% where that bug lives.
%
% Series are (planner, delta) pairs. ALL THREE deltas (`large`/`fine` copied from
% paper_benchmark.cu's own current coarse/fine deltas, plus this sweep's own pre-existing `tiny`)
% run the FULL comparison -- none is restricted to KinoPaxPlus alone (see deltaPlusOnly below and
% DELTA_EXTRA_ARGS in run_countingstars_sweep.sh, all-false this pass):
%
%   KPAX, KinoPaxPlus                                                                   =   2
%   KinoPaxSTARTrue  syclopCap 1.0 (no cap) x ancestorPrune {0, 1}                       =   2
%   CountingStars    ONE FIXED POINT (bufferSlope 1.2, bufferFloor 0.3,
%                    explore_frac 0.1, cost_frac 0.8)                                    =   1
%                                                                                      -----
%                                                                        per delta          5
%                                                                      x 3 deltas     x    3
%                                                                                      -----
%                                                                                           15
%
% KinoPaxSTARCleanCost AND KPAXCap ARE KEPT OUT OF THIS SWEEP -- see the header of
% countingstars_sweep.cu; they are runnable on their own via kinopaxstar_cost_tuning_sweep.cu /
% kinopaxstar_combo_tuning_sweep.cu.
%
% RUN ONCE PER COST METRIC TOO -- effort ONLY this pass (length dropped, see metrics below): the
% series count above applies to this one build.
%
% ALL FIVE SERIES RUN AT ALL THREE DELTAS THIS PASS -- a hang/crash found at one delta is not
% assumed to reproduce at the others, so all three need the full comparison measured, not
% KinoPaxPlus alone at the finer ones.
%
% THE OPTIMAL-ACCEPT-BUDGET TOGGLE IS GONE. It ran as an on/off axis (optimalAcceptBudgeted) in the
% previous pass through this file: the OPTIMAL door (a candidate at distance 0 from its region's
% minimum) used to be UNCAPPED, admitted unconditionally outside any budget. That sweep confirmed
% folding it into the SAME cost-distance histogram/cutoff CHEAPEST already spends against -- it
% always lands in cost bucket 0 (csCostBucket(0.0f, distMax) == 0 for any distMax), and has to
% clear that cutoff/boundary-roll like anything else -- improves final cost / time-to-first-
% solution. It is now the planner's ONLY behavior; there is no toggle, no h_optimalAcceptBudgeted_
% field, and no `_ob` label token left.
%
% CountingStars runs at a fixed point on all four of these axes this pass (bufferSlope/bufferFloor,
% the B ramp; explore_frac/cost_frac, the two shares of B) -- the sweep that tuned them is done;
% see the file header above for why this file's job right now is bug isolation, not tuning:
%
%     x         = itr / fill_iters                             (fraction of the run elapsed)
%     B_frac(x) = bufferSlope * x + bufferFloor
%     B(x)      = floor(B_frac(x) * MAX_TREE_SIZE / fill_iters)  -- RECOMPUTED EVERY ITERATION
%
% B IS A PURE HOST SCALAR (read only inside updateFrontier(), never by propagateFrontier() or any
% device kernel directly), so making it dynamic cost no device array, no new kernel, and no new
% synchronisation -- one floating-point formula recomputed once per iteration. B rides in the data
% as the goal_frontier_size column, which VARIES row to row within a run.
%
% Read the figures in this order:
%
%   0. goal_frontier_size vs iteration     Read FIRST: confirms the realized ramp actually matches
%                               slope*x+floor before reading anything that depends on B.
%   1. frontier_repeat_size / frontier_size    sanity: the realised mean rep. v3.3: fan-out is
%                               door-count (nodeBlocks = popcount(door)), so this should sit very
%                               close to 1 always -- at most 2 doors ever co-fire on one candidate.
%   2. budget_used / goal_frontier_size, AS A CURVE against a MOVING target. See the note on
%                               where B binds.
%   3. optimal_count against admitted_cost   THE OPTIMAL-DOOR STARVATION SIGNAL: optimal_count is
%                               pass 1's measured population at distance 0; admitted_cost is what
%                               pass 2 actually let through the SAME cutoff CHEAPEST uses. Their
%                               gap is how many optimal candidates the budget did not have room for
%                               at a given (bufferSlope, bufferFloor, explore_frac, cost_frac) point.
%   4. admitted_costdist        THE CHEAPEST DOOR'S ACTUAL SHARE, against admitted_explore and
%                               admitted_cost. Pinned at 0 means the cutoff solve is degenerate;
%                               equal to cost_frac * B every iteration means it is working as
%                               designed.
%   5. cost_cutoff_dist / dist_max             whether the log bucket map has the right shape. A
%                               collapse to the 2^-21 floor means every candidate is in bucket 0 and
%                               the door has degraded to a uniform draw among near-optimal
%                               candidates -- switch csCostBucket to linear, a one-line change.
%   6. admitted_both / admitted_opt_fresh_both overlap    0 = the signals are independent; 1 = one
%                               fraction is being spent twice.
%   7. ord_cutoff               rising = regions filling, freshness getting scarce. 0 = explore_frac
%                               inert; 256 = saturated, so explore_frac is not binding either.
%   8. block_scale              near 0 = the rep >= 1 floor ate the budget, fan-out is inert.
%   9. First-solution time and cost, final cost, and -- THE ACTUAL QUESTION THIS PASS ASKS --
%                               whether any series at any delta reproduces paper_benchmark.cu's
%                               cudaErrorIllegalAddress / hang. A crash or a run that never reaches
%                               a final iteration count is the signal, not a cost number.
%
% WHERE B BINDS. NOTHING ON THE CANDIDATE SIDE IS UNCAPPED ANY MORE (v3.4, permanent) -- the
% region-best reactivation GUARANTEE that used to also be uncapped is gone permanently too, folded
% into the reactivation budget (see CS_DOORBIT_GUAR in CountingStars.cuh). So B binds whenever the
% candidate pool for an iteration exceeds it, and is a soft target otherwise.
%
% THAT IS THE POINT, NOT A PROBLEM. B binds EARLY in a run and then stops, at an iteration that
% moves with the WHOLE RAMP SHAPE (bufferSlope and bufferFloor together) rather than a single
% fill_frac, and early is exactly where time-to-first-solution is decided. Read
% budget_used/goal_frontier_size as a CURVE rather than a single number: the iteration where it
% crosses 1 is the measurement, and it should track a MOVING target now that B itself climbs over
% the run.
%
% SCORE FLOOR. Graph's Syclop floor is 1/N_active (the mean share) rather than a fixed
% EPSILON = 1e-2, which exceeded the score it floored by ~270x and capped the number of
% discriminated regions at 1/EPSILON = 100 at ANY grid size. KPAX deliberately keeps the legacy
% floor so it stays an unmodified baseline. COUNTINGSTARS HAS NO SCORE AT ALL -- it never reads
% vertexScores, h_scoreFloor_, h_nActive_ or regionCoverage in any decision -- so it writes NaN
% there and simply does not draw on that panel.
%
% ENCODING THIS PASS: colour + marker + style are assigned PER PLANNER now that CountingStars is a
% single fixed point rather than a grid -- KPAX near-black/'s', KinoPaxPlus blue/'d' (both drawn
% thicker as reference anchors), KinoPaxSTARTrue two shades of crimson/'p'(anc0)/'h'(anc1),
% CountingStars amber/'o'. Line width = delta. Every legend here is CLICKABLE - click an entry to
% hide/show that series.
%
% FAIR-COMPARISON NOTE: an "iteration" is a different unit of work per planner, so
% cost-vs-TIME is the fair cross-planner axis. Error bands and error bars are
% deliberately omitted throughout; the scatter shows run means only.
%
% USAGE: cd into the data directory, then call the script BY NAME, not via run():
%   cd build/Data/Benchmarks/CountingStars/zigzag     % or .../house, .../narrowPassage
%   addpath('<repo>/scripts')
%   process_countingstars_and_plot
% run('<abs path>/process_countingstars_and_plot.m') would cd to the scripts folder
% first, and dataDir below ('' = current folder) would then find nothing.

clear; clc; close all;

%% --- Configuration ---
dataDir = '';   % '' = current directory (run this from Data/Benchmarks/CountingStars/<env>)

% One environment per run — must match the subfolder you cd'd into.
% SCOPE: zigzag this pass (matches ENV_NAMES in run_countingstars_sweep.sh).
% ONE PER RUN -- each environment writes to its own subfolder, so set this to match the folder you
% cd'd into and re-run for another.
%   'zigzag' -> 'Zigzag Corridor',  'narrowPassage' -> 'Narrow Passage',  'house' -> 'House'
environments = {'zigzag'};
envTitles    = {'Zigzag Corridor'};
% environments = {'house'};   envTitles = {'House'};
% environments = {'narrowPassage'};   envTitles = {'Narrow Passage'};

% Cost metric axis — effort ONLY this pass (length dropped; see the header block above).
metrics      = {'effort'};
metricTitles = {'Control Effort'};
metricYLabels = {'Path Cost (control effort)'};

% Delta axis — OVERLAID inside each figure, encoded as line WIDTH. The filename token is
% sprintf('%s_%s', delta, metric), e.g. 'tiny_effort'. THREE deltas this pass -- `large`/`fine`
% copied from paper_benchmark.cu's own current coarse/fine deltas (kept in step by hand; there is
% no cross-check between the two sweep tools), plus this sweep's own pre-existing `tiny` (NOT
% paper's own tiny, which currently hangs on KPAX in an open environment -- see
% run_countingstars_sweep.sh's header). ALL THREE run the full comparison (see deltaPlusOnly
% below), so there is no KinoPaxPlus-only delta here.
deltas      = {'large', 'fine', 'tiny'};
deltaTitles = {'9k', '262k W-refined', '593k V-refined'};
deltaWidths = [1.0, 1.8, 2.6];

% WHICH ARMS EXIST AT EACH DELTA. ALL THREE deltas run the full comparison -- a tuning conclusion
% at one delta is not assumed to hold at the others, so CountingStars and every baseline are
% measured at all three. `--only-kinopaxplus` is lifted off every delta in run_countingstars_sweep.sh
% to match.
%
% MUST MATCH DELTA_EXTRA_ARGS in run_countingstars_sweep.sh: "--only-kinopaxplus" there is a true
% here. When these drift, loadRuns() silently finds no files and reports "0 runs" for the orphaned
% series rather than erroring -- the failure mode that wastes a whole sweep.
% cross_check_countingstars_grid.py asserts it.
deltaPlusOnly = [false, false, false];

deltaLabel = '3 deltas overlaid';

% CountingStars ONE FIXED POINT -- must match CS_BUFFER_SLOPE / CS_BUFFER_FLOOR / CS_EXPLORE_FRAC /
% CS_COST_FRAC in countingstars_sweep.cu (the same point paper_benchmark.cu itself runs at). Values
% are the label tokens exactly as they appear in the filename: round(100x) for slope/floor,
% round(1000x) for explore/cost -- see countingStarsLabel() in the benchmark.
% cross_check_countingstars_grid.py asserts these stay in step with the .cu; when they drift,
% MATLAB reports "0 runs" for the orphaned series rather than erroring.
csBufferSlope = 120;   % bufferSlope 1.2
csBufferFloor = 30;    % bufferFloor 0.3
csExploreFrac = 100;   % explore_frac 0.1
csCostFrac    = 800;   % cost_frac 0.8

% KinoPaxSTARTrue -- two fixed naive points, added to test whether THIS planner, not KPAX, is
% where paper_benchmark.cu's illegal-memory-access/hang bug lives. Must match trueLabel() in
% countingstars_sweep.cu: h_syclopCap_ pinned at 1.0 (no cap, a genuine no-op) at both points;
% only h_ancestorPrune_ varies (0 = pure OR-fusion == stock KinoPaxSTARNoGoalBias, 1 = adds the
% cost-guarded stale-best prune on top).
trueCap                  = 100;    % syclopCap 1.0 -> round(100 * 1.0)
trueAncestorPruneValues  = [0 1];

% --- Build the series arrays: (planner, delta) pairs ---
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
% Each series' bufferFloor token, NaN for anything that is not a CountingStars arm. Used as the
% "is this CountingStars" NaN guard at several panels below. (The colour lookup itself is keyed off
% bufferFloor combined with explore_frac -- see colorIdx in the loop below -- but this array only
% ever needs to distinguish "CountingStars or not", so it stays bufferFloor alone.)
%
% B ITSELF IS NOT CARRIED HERE ANY MORE. v2 had to, because B was a per-run setting and not in the
% data; v3 derives it inside the planner and logs it as the goal_frontier_size COLUMN, so the budget
% figure reads its divisor straight out of the CSV. That removes the last place the plot script had
% to know a piece of the planner's arithmetic.
plannerBufferFloor = [];

% Fixed per-planner colours/markers/styles for the two new KinoPaxSTARTrue points and the
% CountingStars point -- both crimson shades and the amber match paper_benchmark.cu's own
% process_paper_benchmark_and_plot.m so the same planner reads the same way in either tool.
trueColors  = [0.70 0.15 0.20;    % anc0 (naive, no prune)      -- lighter crimson
               0.45 0.05 0.10];   % anc1 (naive, stale-best prune) -- darker crimson
trueStyles  = {'-', '-.'};
trueMarkers = {'p', 'h'};         % pentagram (anc0), hexagram (anc1)
csColor     = [0.85 0.45 0.05];   % CountingStars, single fixed point -- amber

for di = 1:numel(deltas)
    dWidth = deltaWidths(di);
    dTag   = deltaTitles{di};
    dPlus  = deltaPlusOnly(di);    % this delta ran --only-kinopaxplus: no other arm exists here

    if ~dPlus

    % --- KPAX baseline. Gated with the rest: a --only-kinopaxplus delta does not run it. ---
    plannerNames    = [plannerNames,   {'KPAX'}];                                         %#ok<AGROW>
    plannerDisplay  = [plannerDisplay, {sprintf('KPAX [%s]', dTag)}];                     %#ok<AGROW>
    plannerColors   = [plannerColors;  0.10 0.10 0.10];                                   %#ok<AGROW>
    plannerStyles   = [plannerStyles,  {'-'}];                                            %#ok<AGROW>
    plannerMarkers  = [plannerMarkers, {'s'}];                                            %#ok<AGROW>
    plannerWidths   = [plannerWidths,  dWidth + 1.1];                                     %#ok<AGROW>
    plannerBaseline = [plannerBaseline, true];                                            %#ok<AGROW>
    plannerDeltaIdx = [plannerDeltaIdx, di];                                              %#ok<AGROW>
    plannerBufferFloor = [plannerBufferFloor, NaN];                                             %#ok<AGROW>

    % --- KinoPaxSTARTrue: two fixed naive points (anc0, anc1). Added to test whether this
    % planner -- not KPAX -- is where paper_benchmark.cu's illegal-memory-access/hang bug lives. ---
    for ai = 1:numel(trueAncestorPruneValues)
        anc = trueAncestorPruneValues(ai);
        plannerNames{end + 1}   = sprintf('KinoPaxSTARTrue_cap%d_anc%d', trueCap, anc); %#ok<SAGROW>
        plannerDisplay{end + 1} = sprintf('KinoPaxSTARTrue anc%d [%s]', anc, dTag);       %#ok<SAGROW>
        plannerColors(end + 1, :) = trueColors(ai, :);                                    %#ok<SAGROW>
        plannerStyles{end + 1}    = trueStyles{ai};                                       %#ok<SAGROW>
        plannerMarkers{end + 1}   = trueMarkers{ai};                                      %#ok<SAGROW>
        plannerWidths(end + 1)    = dWidth;                                               %#ok<SAGROW>
        plannerBaseline(end + 1)  = false;                                                %#ok<SAGROW>
        plannerDeltaIdx(end + 1)  = di;                                                   %#ok<SAGROW>
        plannerBufferFloor(end + 1) = NaN;                                                %#ok<SAGROW>
    end

    % --- CountingStars: ONE FIXED POINT (matches CS_BUFFER_SLOPE/... in countingstars_sweep.cu,
    % the same point paper_benchmark.cu itself runs at). ---
    plannerNames{end + 1}   = sprintf('CountingStars_bs%d_bf%d_ef%d_cf%d', ...
                                      csBufferSlope, csBufferFloor, csExploreFrac, csCostFrac); %#ok<SAGROW>
    plannerDisplay{end + 1} = sprintf('CountingStars [%s]', dTag); %#ok<SAGROW>
    plannerColors(end + 1, :) = csColor;   %#ok<SAGROW>
    plannerStyles{end + 1}    = ':';       %#ok<SAGROW>
    plannerMarkers{end + 1}   = 'o';       %#ok<SAGROW>
    plannerWidths(end + 1)    = dWidth + 0.8;   %#ok<SAGROW>
    plannerBaseline(end + 1)  = false;          %#ok<SAGROW>
    plannerDeltaIdx(end + 1)  = di;              %#ok<SAGROW>
    plannerBufferFloor(end + 1) = csBufferFloor; %#ok<SAGROW>

    end   % ~dPlus

    % --- KinoPaxPlus. Runs at every delta, same as every other arm this pass -- its advantage is a
    % tiny frontier at a fine discretisation, and a small F is exactly what CountingStars'
    % goal_frontier_size is trying to buy directly, so it is the natural reference at `tiny` too. ---
    plannerNames    = [plannerNames,   {'KinoPaxPlus'}];                                  %#ok<AGROW>
    plannerDisplay  = [plannerDisplay, {sprintf('KinoPaxPlus [%s]', dTag)}];              %#ok<AGROW>
    plannerColors   = [plannerColors;  0.20 0.40 0.80];                                   %#ok<AGROW>
    plannerStyles   = [plannerStyles,  {'--'}];                                           %#ok<AGROW>
    plannerMarkers  = [plannerMarkers, {'d'}];                                            %#ok<AGROW>
    plannerWidths   = [plannerWidths,  dWidth + 1.1];                                     %#ok<AGROW>
    plannerBaseline = [plannerBaseline, true];                                            %#ok<AGROW>
    plannerDeltaIdx = [plannerDeltaIdx, di];                                              %#ok<AGROW>
    plannerBufferFloor = [plannerBufferFloor, NaN];                                             %#ok<AGROW>
end

numRunsPer = 50 * ones(1, numel(plannerNames));   % max runs searched (missing files skipped)

MAX_FLOAT_THRESH = 1e30;   % best_cost sentinel (MAX_FLOAT / INFINITY) -> NaN
numTimeSamples   = 500;

% Reference line for the tree-growth panel. MUST match the MAX_TREE_SIZE / MAX_ITER that
% run_countingstars_sweep.sh writes into config.h -- neither is in the CSV, so this is the one place
% the plot has to be told. Only used to draw the dashed reference; nothing else depends on it.
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
        % score_floor should sit flat at EPSILON = 0.01 for KPAX, which deliberately keeps the
        % legacy floor so it stays an unmodified baseline. NEITHER COUNTINGSTARS NOR KINOPAXPLUS
        % DRAWS ON THE LEFT PANEL: CountingStars has no Syclop score at all, and KinoPaxPlus has no
        % Graph -- both write NaN there. CountingStars DOES write cost_scale (right panel) -- the
        % same global denominator CleanCost's costProbExpGlobal originated, now the denominator of
        % a CountingStars candidate's distance. Series lacking either column are skipped by
        % getCol/plotMeanTime.
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

        %% ---------- FIGURE: THE REALIZED BUDGET RAMP ----------
        % READ THIS ONE FIRST, before anything else that depends on B. This is the direct visual
        % check that the realized B(itr) actually matches the intended slope*x + bufferFloor before
        % reading any panel that divides by it.
        %
        % One CountingStars series per delta now (fixed point, not a grid): each should rise
        % roughly linearly from bufferFloor's starting value toward (bufferSlope + bufferFloor) *
        % MAX_TREE_SIZE / fill_iters at the last iteration.
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Budget Ramp (%s)', envTitle, costTitle), ...
               'Position', [100 100 900 560]);
        hold on;
        for pi = 1:nPlanner
            if isnan(plannerBufferFloor(pi)), continue; end   % not a CountingStars series
            plotMeanIter(R{pi}, @(t) getCol(t, 'goal_frontier_size'), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        grid on;
        xlabel('Iteration'); ylabel('goal\_frontier\_size (B)');
        title({'The realized budget ramp', 'rising = the ramp in effect, tracking slope*x + floor'});
        clickableLegend();

        %% ---------- FIGURE: IS THE BUDGET MET ----------
        % THE CLAIM THE WHOLE DESIGN RESTS ON, and therefore the first figure to read. B is an
        % INPUT, not a cap: the doors fill it in priority order and F is supposed to come out at B
        % by construction. Everything else on this grid is a tuning question; this one is a
        % correctness question.
        %
        %   budget_used / B == 1   the budget is met.
        %   budget_used / B  < 1   SHORTFALL -- a door is not filling its share. Read the door panel
        %                          on the right to see which one ran dry.
        %   budget_used / B  > 1   OVERSHOOT. Nothing on the candidate side is uncapped any more
        %                          (v3.4, permanent), so a genuine overshoot here is unexpected and
        %                          worth investigating directly against the cost-cutoff figure
        %                          below, not a known regime this grid is on to exercise.
        %
        % SO READ THIS AS A CURVE, NOT A NUMBER. B binds EARLY in a run and then stops, at an
        % iteration that moves with the WHOLE RAMP SHAPE. The iteration where the curve crosses 1
        % IS the measurement -- early is exactly where time-to-first-solution is decided.
        %
        % B COMES OUT OF THE DATA. It is the goal_frontier_size column, written by the planner that
        % derived it, so this divides by what the run actually used rather than by what the label
        % implies.
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Budget vs Doors (%s)', envTitle, costTitle), ...
               'Position', [120 120 1400 620]);
        subplot(1, 2, 1); hold on;
        for pi = 1:nPlanner
            if isnan(plannerBufferFloor(pi)), continue; end   % not a CountingStars series
            plotMeanIter(R{pi}, @(t) safeRatio(getCol(t, 'budget_used'), ...
                                               getCol(t, 'goal_frontier_size')), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        set(gca, 'YScale', 'log'); grid on;
        yline(1, 'k--', 'budget met', 'LineWidth', 1.4, 'HandleVisibility', 'off');
        xlabel('Iteration'); ylabel('budget\_used / goal\_frontier\_size');
        title({'Is the budget met?', ...
               'below 1 = a door ran dry; above 1 = unexpected -- nothing here is uncapped any more'});
        clickableLegend();

        % --- Which door filled it. Every node came through a named door and the counts are exact,
        % so a shortfall on the left has an address here.
        %
        %   optimal_count      pass 1's MEASURED population at distance 0 -- "how many optimal
        %                      candidates existed", not "how many were let in" (that is
        %                      admitted_cost, below).
        %   admitted_cost      v3.4: pass 2's ADMITTED count via CS_DOORBIT_OPTIMAL, through the
        %                      SAME cutoff CHEAPEST uses (permanent, not a toggle). STRICTLY LESS
        %                      than optimal_count whenever the budget starves some of them -- the
        %                      gap between the two curves is that starvation count. Drawn as a
        %                      dotted line right next to optimal_count's solid one, same color, so
        %                      the gap reads directly.
        %   admitted_explore   the freshness door, spending explore_frac * B.
        %   admitted_costdist  spending cost_frac * B on the smallest cost distances -- also where
        %                      OPTIMAL candidates now always compete (see CS_DOORBIT_OPTIMAL).
        %                      Pinned at 0 while cost_frac > 0 means the cutoff solve is degenerate;
        %                      sitting at cost_frac * B every iteration means it works as designed.
        %   reactivated_best   ALWAYS 0 now -- the region-best reactivation GUARANTEE was folded
        %                      permanently into the budgeted reactivation histogram in an earlier
        %                      pass (see CS_DOORBIT_GUAR in CountingStars.cuh). Column kept for CSV
        %                      schema stability, not because the arm still fires.
        %   reactivated_cost   the CHEAPEST reactivation arm, spending the WHOLE react_frac * B
        %                      budget on the cheapest dormant nodes (including former-GUARANTEE
        %                      candidates, now folded in here) -- this should carry essentially all
        %                      of Part B's volume.
        %   reactivated_count  the COMPLETENESS FLOOR alone -- ~ react_floor * dormant_count, so
        %                      ~30 nodes. LARGE HERE MEANS the floor is doing reach work it was not
        %                      sized for.
        %   admitted_floor     the ADMISSION completeness floor's yield -- fires only for a
        %                      candidate that cleared none of OPTIMAL/FRESHEST/CHEAPEST. Should
        %                      track ~accept_floor * candidates_this_iteration; large here means the
        %                      floor is doing reach work rather than plugging the gap.
        %
        % The `reactivated` column is all THREE Part B arms, so it should equal
        % reactivated_best + reactivated_cost + reactivated_count -- trivially true now that
        % reactivated_best is always 0, but the identity is unchanged in form.
        %
        % OPTIMAL, EXPLORE AND COSTDIST ALL OVERLAP: admitted_explore and admitted_costdist are a
        % union over one candidate pool; OPTIMAL also competes for FRESHEST (never CHEAPEST -- see
        % CS_DOORBIT_OPTIMAL). The full identity, using what pass 2 actually ADMITTED
        % (admitted_cost, not optimal_count -- see above):
        % admitted == admitted_cost + admitted_explore + admitted_costdist + admitted_floor
        %           - admitted_opt_fresh_both - admitted_both, and both overlaps have their own
        % panel below.
        subplot(1, 2, 2); hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) getCol(t, 'optimal_count'), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
            plotMeanIter(R{pi}, @(t) getCol(t, 'admitted_cost'), ...
                         plannerColors(pi, :), ':', plannerWidths(pi), '');
            plotMeanIter(R{pi}, @(t) getCol(t, 'admitted_explore'), ...
                         plannerColors(pi, :), '--', max(0.5, plannerWidths(pi) - 0.6), '');
            plotMeanIter(R{pi}, @(t) getCol(t, 'admitted_costdist'), ...
                         plannerColors(pi, :), '-', max(0.5, plannerWidths(pi) - 0.8), '');
            plotMeanIter(R{pi}, @(t) getCol(t, 'reactivated_best'), ...
                         plannerColors(pi, :), '-.', max(0.5, plannerWidths(pi) - 0.3), '');
            plotMeanIter(R{pi}, @(t) getCol(t, 'reactivated_cost'), ...
                         plannerColors(pi, :), '--', max(0.5, plannerWidths(pi) - 0.9), '');
            plotMeanIter(R{pi}, @(t) getCol(t, 'reactivated_count'), ...
                         plannerColors(pi, :), ':', max(0.5, plannerWidths(pi) - 0.6), '');
            plotMeanIter(R{pi}, @(t) getCol(t, 'admitted_floor'), ...
                         plannerColors(pi, :), ':', max(0.5, plannerWidths(pi) - 0.3), '');
        end
        set(gca, 'YScale', 'log'); grid on;
        xlabel('Iteration'); ylabel('nodes');
        title({'optimal measured (thick solid) vs admitted (thick dotted), explore (dashed), cheapest (thin solid),', ...
               'guarantee -- always 0 now (dash-dot), cheap-reactivation (thin dashed), reactivation floor (dotted), admission floor (thin dotted)'});

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
        figure('Name', sprintf('%s - Selection Cutoffs (%s)', envTitle, costTitle), ...
               'Position', [140 140 1700 600]);
        subplot(1, 3, 1); hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) getCol(t, 'ord_cutoff'), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        grid on;
        yline(256, 'k--', 'saturated (all fresh)', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        xlabel('Iteration'); ylabel('ord\_cutoff (region population)');
        title({'Freshness cutoff', 'rising = regions filling; 0 = explore\_frac inert; 256 = not binding'});
        clickableLegend();

        % --- THE COST DOOR'S CUTOFF, AND WHETHER ITS BUCKET MAP HAS THE RIGHT SHAPE. This is the
        % panel that decides whether v3's new door works.
        %
        % cost_cutoff_dist is the DISTANCE threshold, not the bucket index -- the index is only
        % meaningful against the dist_max that produced it and dist_max moves every iteration, so
        % the distance is the only version comparable across a run. Both are drawn, normalised by
        % dist_max so the ratio is readable on one axis:
        %
        %   ratio near 1        the door is admitting almost everything; cost_frac is not binding.
        %   ratio in 1e-3..1    healthy. The log buckets are resolving the distribution and the
        %                       cutoff is landing somewhere inside it.
        %   ratio near 2^-21    EVERYTHING IS IN BUCKET 0. The distances have piled up below the
        %                       bottom of the 21-octave window, so the boundary roll is choosing a
        %                       uniform random subset of near-optimal candidates and the door has
        %                       stopped discriminating. THE FIX IS A ONE-LINE CHANGE to csCostBucket
        %                       in include/planners/CountingStars.cuh -- a linear map over
        %                       [0, distMax] instead of the log one.
        subplot(1, 3, 2); hold on;
        for pi = 1:nPlanner
            if isnan(plannerBufferFloor(pi)), continue; end   % not a CountingStars series
            plotMeanIter(R{pi}, @(t) safeRatio(getCol(t, 'cost_cutoff_dist'), ...
                                               getCol(t, 'dist_max')), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        set(gca, 'YScale', 'log'); grid on;
        yline(2^-21, 'k--', 'bucket 0 floor (door degenerate)', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        xlabel('Iteration'); ylabel('cost\_cutoff\_dist / dist\_max');
        title({'CANDIDATE cost-distance cutoff, as a fraction of the anchor', ...
               'at the floor = every candidate in bucket 0, door degenerate'});
        clickableLegend();

        % --- v3.1: THE SAME READING FOR PART B'S COST ARM, over dormant tree nodes rather than
        % candidates. This is the panel that says whether cost-selective reactivation is actually
        % selecting.
        %
        % dist_max is the CANDIDATE anchor, reused -- a dormant node above it clamps into the top
        % bucket, which is harmless while the cutoff sits below it (the arm takes the SMALLEST
        % distances, and everything in the top bucket is the expensive tail being excluded).
        %
        %   AT 1 (the top bucket)   the budget exceeds the population below dist_max, so the arm is
        %                           partly selecting at RANDOM within the clamped tail. This is the
        %                           one case where reusing the candidate anchor bites, and the fix
        %                           is a separate tree-side anchor.
        %   WELL BELOW 1            healthy: the cutoff is landing inside the resolved range and the
        %                           arm is genuinely picking the cheapest dormant nodes.
        subplot(1, 3, 3); hold on;
        for pi = 1:nPlanner
            if isnan(plannerBufferFloor(pi)), continue; end   % not a CountingStars series
            plotMeanIter(R{pi}, @(t) safeRatio(getCol(t, 'react_cutoff_dist'), ...
                                               getCol(t, 'dist_max')), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        set(gca, 'YScale', 'log'); grid on;
        yline(1, 'k--', 'top bucket (anchor exceeded)', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        xlabel('Iteration'); ylabel('react\_cutoff\_dist / dist\_max');
        title({'REACTIVATION cost cutoff (dormant tree nodes)', ...
               'at 1 = budget exceeds the population below the anchor'});
        clickableLegend();

        %% ---------- FIGURE: DO THE SELECTION DOORS BUY DIFFERENT THINGS ----------
        % The freshness and cheapness doors select over the SAME candidate pool on independent
        % signals, so their picks overlap and admitted_both counts the overlap exactly. v3.3: OPTIMAL
        % now also competes for FRESHEST, so it has its own overlap term, admitted_opt_fresh_both --
        % plotted alongside as a second curve per series (solid = FRESHEST/CHEAPEST, dotted =
        % OPTIMAL/FRESHEST). OPTIMAL never overlaps CHEAPEST (see CS_DOORBIT_OPTIMAL), so there is no
        % third term.
        %
        %   near 0   the two signals are independent, which is the case worth having: each door is
        %            buying nodes the other would not have.
        %   near 1   one door is re-admitting what the other already took, so one fraction is being
        %            spent twice on the same nodes and the budget it was given is going to the draw
        %            (or, for the OPTIMAL/FRESHEST curve, is simply inert) instead.
        %
        % Note both can only be read where their inputs are non-zero; safeRatio returns NaN on a
        % zero denominator, which the ablation arms and any FRESHEST=OPTIMAL-only iteration hit.
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Door Overlap (%s)', envTitle, costTitle), ...
               'Position', [160 160 900 560]);
        hold on;
        for pi = 1:nPlanner
            if isnan(plannerBufferFloor(pi)), continue; end   % not a CountingStars series
            plotMeanIter(R{pi}, @(t) safeRatio(getCol(t, 'admitted_both'), ...
                                               sumCols(t, 'admitted_explore', 'admitted_costdist')), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
            plotMeanIter(R{pi}, @(t) safeRatio(getCol(t, 'admitted_opt_fresh_both'), ...
                                               sumCols(t, 'optimal_count', 'admitted_explore')), ...
                         plannerColors(pi, :), ':', max(0.5, plannerWidths(pi) - 0.6), '');
        end
        grid on; ylim([0 1]);
        xlabel('Iteration'); ylabel('overlap / union');
        title({'Selection-door overlap: FRESHEST/CHEAPEST (main) vs OPTIMAL/FRESHEST (dotted)', ...
               '0 = the two signals are independent; 1 = one fraction is being spent twice'});
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

        % --- 2. How fast is the tree actually filling? ---
        % THERE IS NO GROWTH CONTROLLER IN v2 -- the dashed line is a REFERENCE, not a target the
        % planner is tracking. Tree growth is an OUTPUT here: it is however many candidates the
        % doors admitted, and the budget governs the FRONTIER, not the tree. Read a shortfall as
        % "the candidate pool ran dry" (propagation is colliding, or F is too small to produce
        % enough candidates), never as "the controller is behind schedule". ---
        subplot(1, 3, 2); hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) getCol(t, 'tree_size'), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        % plot([0 growthIters], [0 maxTreeSize], 'k--', 'LineWidth', 1.2, ...
        %      'DisplayName', 'fill MAX\_TREE\_SIZE by MAX\_ITER');
        xlabel('Iteration'); ylabel('tree\_size'); grid on;
        title({'Tree growth against a linear fill reference', ...
               'an OUTPUT, not a target -- v2 budgets the frontier, not the tree'});

        % --- 3. What is the frontier actually made of? ---
        % `reactivated` counts frontier bits among the PRE-EXISTING tree, i.e. exactly Part B's
        % output, so this is the share of the frontier that is re-expansion rather than new nodes.
        %
        % READ THIS PANEL FIRST WHEN KERNEL1 FAILS EARLY. Part B's reactivation is entirely budgeted
        % now (the region-best GUARANTEE that used to be unconditional was folded permanently into
        % the cheapest-reactivation share of react_frac * B -- see CS_DOORBIT_GUAR), so F has no
        % structural floor from an uncovered-region guarantee any more; a floor here comes only from
        % the reactivation completeness floor (reactivated_count) plus whatever reactivated_cost
        % actually admits. Since rep >= 1, frontierRepeatSize >= F, and kernel2 is forced once
        % 32*F > remaining whatever the budget says. A curve pinned near 100% means F is
        % reactivation-dominated: the admissions (optimal + explore + costdist) are a rounding error
        % next to Part B, and B is being met by re-expansion rather than by new ground.
        subplot(1, 3, 3); hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) 100 * safeRatio(getCol(t, 'reactivated'), ...
                                                     getCol(t, 'frontier_size')), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        ylim([0 105]);
        xlabel('Iteration'); ylabel('% of frontier from Part B'); grid on;
        title({'Frontier composition: reactivated / frontier\_size', 'near 100% = Part B (now entirely budgeted) dominates F'});
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
        %   block_scale = 1     the buffer ceiling never bound; door-count fan-out
        %                       (nodeBlocks = popcount(door), set at admission) is unscaled.
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
        % The cost OF the first solution, as distinct from the final cost. best_cost is already the
        % RUNNING best, so this is simply its value at the first finite row -- and because it is
        % monotone non-increasing, mFirstSolCost >= mFinalCost for every series. The gap between the
        % two IS that variant's refinement gain after it first reached the goal.
        mFirstSolCost = NaN(1, nPlanner);
        mFinalCost    = NaN(1, nPlanner);
        mTotalTime    = NaN(1, nPlanner);
        mSuccess      = NaN(1, nPlanner);
        for pi = 1:nPlanner
            runs = R{pi};
            fiVals = []; fstVals = []; ftsVals = []; fcVals = []; ttVals = []; fscVals = [];
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
                fsc = firstSolCost(runs{ri}, MAX_FLOAT_THRESH);
                if ~isnan(fsc), fscVals(end + 1) = fsc; end %#ok<SAGROW>
                fc = finalCost(runs{ri}, MAX_FLOAT_THRESH);
                if ~isnan(fc), fcVals(end + 1) = fc; end %#ok<SAGROW>
                ttVals(end + 1) = runs{ri}.elapsed_time_ms(end) / 1000; %#ok<SAGROW>
            end
            if ~isempty(fiVals),  mFirstIter(pi)    = mean(fiVals);  end
            if ~isempty(fstVals), mFirstSolTime(pi) = mean(fstVals); end
            if ~isempty(ftsVals), mFirstSolTree(pi) = mean(ftsVals); end
            if ~isempty(fscVals), mFirstSolCost(pi) = mean(fscVals); end
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

        sgtitle(sprintf('Bug-Isolation Comparison at %s \x2014 %s, %s (run means)', ...
                deltaLabel, envTitle, costTitle), 'FontSize', 12, 'FontWeight', 'bold');

        %% ---------- FIGURE: Solution Success Rate ----------
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Success Rate (%s)', envTitle, costTitle), ...
               'Position', [100 100 1300 560]);
        plannerBar(mSuccess, plannerDisplay, plannerColors, 'Success Rate (%)', ...
            sprintf('Solution Success Rate \x2014 %s, %s, %s', envTitle, deltaLabel, costTitle));
        ylim([0 110]);

        %% ---------- FIGURES: Tuning tradeoff scatters (a PAIR, on matched axes) ----------
        % Both have the same x -- mean time to first solution -- and the same markers. Only the y
        % metric differs:
        %
        %   this one    mean FINAL best cost      what the run converged to
        %   the next    mean FIRST-solution cost  what it got the moment it first reached the goal
        %
        % THE Y-LIMITS ARE SHARED, computed once over the finite entries of BOTH metrics and applied
        % to both figures. That is what makes the pair readable rather than two unrelated pictures:
        % best_cost is monotone non-increasing, so every variant sits at or BELOW its first-solution
        % marker here, and the VERTICAL GAP BETWEEN THE TWO FIGURES IS THAT VARIANT'S REFINEMENT
        % GAIN after it first reached the goal. Letting the two autoscale independently destroys
        % exactly that comparison.
        %
        % Lower-left is the winning corner in both (fast AND cheap).
        costLims = [mFirstSolCost(:); mFinalCost(:)];
        costLims = costLims(isfinite(costLims));
        if numel(costLims) >= 2 && max(costLims) > min(costLims)
            pad      = 0.05 * (max(costLims) - min(costLims));
            costYLim = [min(costLims) - pad, max(costLims) + pad];
        else
            costYLim = [];   % nothing solved, or a single value: let MATLAB autoscale
        end

        % The marker legend, written once and used by both titles so they cannot drift apart.
        % marker is assigned per planner now that CountingStars is a single fixed point:
        % \x25a1 KPAX, \x25c7 KinoPaxPlus, \x2606 KinoPaxSTARTrue anc0, \x2605 KinoPaxSTARTrue anc1
        % (hollow vs. filled star, matching the pentagram/hexagram marker shapes), \x25cb
        % CountingStars. plannerDisplay's legend text always carries the exact series identity
        % regardless, so nothing is ever ambiguous.
        % sprintf, NOT a bare concatenation: the \x.... marker glyphs and the \\_ TeX underscore
        % escapes are only resolved by a formatting call, and this string is substituted into the
        % titles below via %s -- which inserts it verbatim rather than re-interpreting it. Built as
        % a plain [...] it would print the escape sequences literally.
        markerKey = sprintf(['lower-left is better (fast and cheap); \x25a1 KPAX, \x25c7 KinoPaxPlus, ' ...
                             '\x2606 True(anc0), \x2605 True(anc1), \x25cb CountingStars']);

        figNum = figNum + 1;
        figure('Name', sprintf('%s - Tradeoff Scatter, Final Cost (%s)', envTitle, costTitle), ...
               'Position', [130 140 1180 700]);
        tradeoffScatter(mFirstSolTime, mFinalCost, plannerMarkers, plannerColors, ...
                        plannerBaseline, plannerDisplay, costYLim);
        xlabel('Avg Time to First Solution (ms)'); ylabel(sprintf('Avg Final %s', costYLab));
        title(sprintf(['Tuning Tradeoff: Time to First Solution vs FINAL Cost \x2014 %s, %s\n%s'], ...
                       envTitle, costTitle, markerKey), 'FontWeight', 'bold');

        %% ---------- FIGURE: the same scatter against FIRST-SOLUTION cost ----------
        % Same x, same markers, same y-limits as the figure above -- so flipping between the two
        % reads directly as how much each variant's cost improved AFTER it first reached the goal.
        %
        % This is the panel that separates the two ways a planner can win on final cost: finding a
        % good path immediately (low here) versus refining a mediocre one (large gap between the two
        % figures). CountingStars' advantage is expected to be the second, since Part B's whole
        % reactivation budget is cost-weighted (spent on the cheapest dormant nodes) and refinement
        % is what Part B does -- KPAX has no such mechanism, and KinoPaxPlus's is unweighted.
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Tradeoff Scatter, First-Solution Cost (%s)', envTitle, costTitle), ...
               'Position', [160 170 1180 700]);
        tradeoffScatter(mFirstSolTime, mFirstSolCost, plannerMarkers, plannerColors, ...
                        plannerBaseline, plannerDisplay, costYLim);
        xlabel('Avg Time to First Solution (ms)');
        ylabel(sprintf('Avg %s of the FIRST Solution', costYLab));
        title(sprintf(['Tuning Tradeoff: Time to First Solution vs FIRST-SOLUTION Cost ' ...
                       '\x2014 %s, %s\n%s'], envTitle, costTitle, markerKey), 'FontWeight', 'bold');
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
                % CountingStars and KinoPaxSTARTrue both use their planner label directly as the
                % filename token, e.g. CountingStars_bs120_bf30_ef100_cf800 or
                % KinoPaxSTARTrue_cap100_anc0 -- mirrors writePerIterationCSV()'s own dispatch in
                % countingstars_sweep.cu (rfind("CountingStars",0)==0 || rfind("KinoPaxSTAR",0)==0).
                % KinoPaxSTARCleanCost and KPAXCap are gone from this sweep (see the file header).
                % The list is a WHITELIST on purpose -- an unrecognised label is a typo or a grid
                % that drifted, and erroring here is far better than silently loading nothing and
                % reporting "0 runs" for a series that was actually written.
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

function c = firstSolCost(tbl, thresh)
    % The cost OF the first solution: best_cost at the first finite (< thresh) row. best_cost is
    % already the RUNNING best, so no extra bookkeeping is needed -- this is the same one-line
    % pattern as firstSolIter / firstSolTime / firstSolTreeSize, over a different column.
    % NaN if the run never found a solution.
    solIdx = find(tbl.best_cost < thresh, 1, 'first');
    if isempty(solIdx), c = NaN; else, c = tbl.best_cost(solIdx); end
end

function tradeoffScatter(x, y, markers, colors, isBaseline, labels, yLim)
    % The shared body of the two tradeoff scatters, so the pair cannot drift in marker size,
    % colour or legend behaviour -- the only thing that differs between them is the y metric.
    hold on;
    for pi = 1:numel(x)
        if isnan(x(pi)) || isnan(y(pi)), continue; end   % never solved -> nothing to place
        % Explicit flag, not a positional guess: the baselines are not simply the last two series
        % (each delta contributes its own KPAX/KinoPaxPlus pair).
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
    % baselines write NaN for every CountingStars-only column, so those series simply do not draw
    % rather than erroring.
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

function v = sumCols(tbl, colA, colB)
    % Elementwise sum of two columns, returning [] when EITHER is absent rather than letting
    % [] + vector reach the addition. getCol's contract is that a missing column comes back empty,
    % so a caller that adds two of its results has to re-establish that contract itself.
    a = getCol(tbl, colA);
    b = getCol(tbl, colB);
    if isempty(a) || isempty(b), v = []; return; end
    n = min(numel(a), numel(b));
    v = a(1:n) + b(1:n);
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
