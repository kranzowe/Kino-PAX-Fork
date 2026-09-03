%% CountingStars v3.2 Sweep Visualization - the buffer becomes a per-iteration ramp
% Reads per-iteration CSVs produced by examples/gpu/countingstars_sweep.cu
% (run via scripts/run_countingstars_sweep.sh).
%
% Series are (planner, delta) pairs. THIS PASS runs the coarse delta and the house environment
% only for the tuned arms; the two finer deltas run KinoPaxPlus alone (see deltaPlusOnly below and
% DELTA_EXTRA_ARGS in run_countingstars_sweep.sh):
%
%   CountingStars         bufferSlope {0, 1.0, 1.5} x bufferFloor {0, 0.1, 0.2}
%                         explore_frac=0.3, cost_frac=0.3 FIXED (not swept this pass)  =  9
%   KinoPaxSTARCleanCost  r2 OFF, w 0.9, k 1, cap 0.03  (one tuned reference point)     =  1
%   KPAXCap               cap {0.03}                                                   =  1
%   KPAX, KinoPaxPlus                                                                  =  2
%                                                                                      -----
%                                                                  at the coarse delta    13
%   KinoPaxPlus at the two finer deltas                                                +  2
%                                                                                      -----
%                                                                                         15
%
% CountingStars runs at the COARSE delta only -- that is what the discretization factor in the
% grid above means, and it is enforced by DELTA_EXTRA_ARGS in run_countingstars_sweep.sh
% (--only-kinopaxplus at the two finer deltas), mirrored by deltaPlusOnly below.
%
% WHAT THIS SWEEP IS ASKING. v3's sweep showed the standard explore-vs-refine tradeoff: a small
% constant buffer (fill_frac = 0.25) found a first solution fast but converged to a worse final
% cost; a large one (0.75) was the reverse. Rather than pick one point on that tradeoff, v3.2 makes
% the buffer VARY over the run:
%
%     x         = itr / MAX_ITER                              (fraction of the run elapsed)
%     B_frac(x) = bufferSlope * x + bufferFloor
%     B(x)      = floor(B_frac(x) * MAX_TREE_SIZE / MAX_ITER)  -- RECOMPUTED EVERY ITERATION
%
% bufferSlope = 0 REPRODUCES v3's CONSTANT B EXACTLY (B_frac(x) = bufferFloor for every x), so that
% subgrid is a free, structural comparison against the old fixed-buffer design, not a separate
% baseline swept again. explore_frac/cost_frac are FIXED at 0.3 each this pass (not swept) to
% isolate the ramp's own effect.
%
% B IS A PURE HOST SCALAR (read only inside updateFrontier(), never by propagateFrontier() or any
% device kernel directly), so making it dynamic cost no device array, no new kernel, and no new
% synchronisation -- one floating-point formula recomputed once per iteration. B rides in the data
% as the goal_frontier_size column, now genuinely VARYING row to row within a run rather than
% constant -- the first pass where plotting it against iteration is worth its own panel.
%
% Read the figures in this order:
%
%   0. goal_frontier_size vs iteration     NEW THIS PASS, and read FIRST: confirms the realized
%                               ramp actually matches slope*x+floor before reading anything that
%                               depends on B. A flat line at every bufferSlope=0 series is the
%                               direct sanity check that the mechanism is wired correctly.
%   1. frontier_repeat_size / frontier_size    sanity: the realised mean rep. It should sit
%                               near 1 with a small excess from thin regions and the both-doors
%                               boost, not near maxBlocks.
%   2. budget_used / goal_frontier_size, AS A CURVE against a now-MOVING target. See the note on
%                               where B binds.
%   3. admitted_costdist        THE CHEAPEST DOOR'S ACTUAL SHARE, against admitted_explore and
%                               optimal_count. Pinned at 0 means the cutoff solve is degenerate;
%                               equal to cost_frac * B every iteration means it is working as
%                               designed.
%   4. cost_cutoff_dist / dist_max             whether the log bucket map has the right shape. A
%                               collapse to the 2^-21 floor means every candidate is in bucket 0 and
%                               the door has degraded to a uniform draw among near-optimal
%                               candidates -- switch csCostBucket to linear, a one-line change.
%   5. admitted_both overlap    0 = the two selection signals are independent (both doors buying
%                               something); 1 = one fraction is being spent twice.
%   6. ord_cutoff               rising = regions filling, freshness getting scarce. 0 = explore_frac
%                               inert; 256 = saturated, so explore_frac is not binding either.
%   7. block_scale              near 0 = the rep >= 1 floor ate the budget, fan-out is inert.
%   8. First-solution time and cost, final cost (figures 11/12)   THE ACTUAL QUESTION: does a ramp
%                               beat the best bufferSlope=0 point on time-to-first-solution AND
%                               close the final-cost gap against CleanCost.
%
% (bufferSlope, bufferFloor) = (0, 0) IS THE DEEPEST CONTROL and is drawn thicker at every
% bufferFloor row: a constant B = 0 (floored to 1), so the three budgeted doors admit nothing every
% iteration and the frontier is optimal + guarantee + a trickle draw. If nothing beats it, none of
% the three budgeted doors is earning its share.
%
% WHERE B BINDS -- AND EVERY B ON THIS GRID IS BELOW THE THRESHOLD. Two doors are uncapped and BOTH
% are bounded by NUM_R1_REGIONS rather than by B: OPTIMAL (at most one region best per region per
% iteration) and GUARANTEE (at most one node per uncovered region). So B stops binding once nActive
% passes it, and every point on this ramp's range stays under the coarse delta's 27,000.
%
% THAT IS THE POINT, NOT A PROBLEM. B binds EARLY in a run and then stops, at an iteration that now
% moves with the WHOLE RAMP SHAPE (bufferSlope and bufferFloor together) rather than a single
% fill_frac, and early is exactly where time-to-first-solution is decided. Read
% budget_used/goal_frontier_size as a CURVE rather than a single number: the iteration where it
% crosses 1 is the measurement, and a late-run overshoot is expected at every point -- more so now,
% since B itself is climbing over the run. This is also what "tree growth is less controlled once
% min cost is always accepted" amounts to.
%
% If the bufferSlope=0 curves are indistinguishable from the ramped ones even early, that is direct
% evidence that capping the guarantee (KinoPaxPlus's hysteresis is the precedent -- un-prune a
% region best only after ~5 idle iterations) is the next lever, not a different ramp.
%
% SCORE FLOOR. Graph's Syclop floor is 1/N_active (the mean share) rather than a fixed
% EPSILON = 1e-2, which exceeded the score it floored by ~270x and capped the number of
% discriminated regions at 1/EPSILON = 100 at ANY grid size. OPT-IN: KPAXCap and CleanCost take it,
% KPAX deliberately keeps the legacy floor so it stays an unmodified baseline. COUNTINGSTARS HAS NO
% SCORE AT ALL -- it never reads vertexScores, h_scoreFloor_, h_nActive_ or regionCoverage in any
% decision -- so it writes NaN there and simply does not draw on that panel.
%
% ENCODING: colour = bufferFloor (near-black smallest -> pale largest); line style = bufferSlope
% (solid = bufferSlope 0, v3's constant-B control); scatter marker = 'o' (fixed -- explore_frac and
% cost_frac no longer vary, so there is no third axis to give its own marker); line width = delta.
% maxBlocks is held at 4, unchanged from v3. CleanCost is crimson, KPAXCap grey-green,
% KPAX near-black, KinoPaxPlus blue -- all four drawn thicker as reference anchors. Every legend
% here is CLICKABLE - click an entry to hide/show that series.
%
% FAIR-COMPARISON NOTE: an "iteration" is a different unit of work per planner, so
% cost-vs-TIME is the fair cross-planner axis. Error bands and error bars are
% deliberately omitted throughout; the scatter shows run means only.
%
% USAGE: cd into the data directory, then call the script BY NAME, not via run():
%   cd build/Data/Benchmarks/CountingStars/house      % or .../zigzag, .../narrowPassage
%   addpath('<repo>/scripts')
%   process_countingstars_and_plot
% run('<abs path>/process_countingstars_and_plot.m') would cd to the scripts folder
% first, and dataDir below ('' = current folder) would then find nothing.

clear; clc; close all;

%% --- Configuration ---
dataDir = '';   % '' = current directory (run this from Data/Benchmarks/CountingStars/<env>)

% One environment per run — must match the subfolder you cd'd into.
%   'zigzag' -> 'Zigzag Corridor',  'house' -> 'House'
% SCOPE: house this pass (matches ENV_NAMES in run_countingstars_sweep.sh).
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

% CountingStars v3.2 grid - must match BUFFER_SLOPES / BUFFER_FLOORS / EXPLORE_FRACS / COST_FRACS
% in countingstars_sweep.cu. Values are the label tokens exactly as they appear in the filenames:
% bufferSlope/bufferFloor as round(100 x float), the two shares as round(1000 x float).
% cross_check_countingstars_grid.py asserts these stay in step with the .cu and the .sh; when they
% drift, MATLAB reports "0 runs" for the orphaned series rather than erroring, which is the failure
% mode that silently wastes a whole sweep.
%
% B IS A RAMP, RECOMPUTED EVERY ITERATION, and it is a CSV COLUMN. The planner computes
%
%     x = itr/MAX_ITER,   B(x) = floor((bufferSlope*x + bufferFloor) * MAX_TREE_SIZE / MAX_ITER)
%
% -- v3's single fill_frac is gone; bufferSlope/bufferFloor together replace it, with
% bufferSlope = 0 reproducing v3's constant B exactly (B(x) = bufferFloor for every x). B travels
% in the data as goal_frontier_size, which NOW GENUINELY VARIES ROW TO ROW within a run instead of
% being constant -- see the new "goal_frontier_size vs iteration" figure below, which did not exist
% under v3 because that column was always a flat line not worth its own panel.
%
% KinoPaxPlus divides the whole budget over a frontier its pruning keeps tiny
% (bf = MAX_TREE_SIZE/(F*32), 40,000 propagations per node at F = 10), which is the number
% prop_attempted/frontier_size is read against.
%
% csExploreFracs / csCostFracs ARE round(1000 x frac) TOKENS, not 100x -- see countingStarsLabel()
% in the benchmark. FIXED AT 0.3 EACH THIS PASS (single-element arrays, not swept) -- v3.2 isolates
% the ramp's own effect by holding them still, the same discipline csMaxBlocks already uses to stay
% held without a shape change to the loop below.
%
% csBufferSlopes / csBufferFloors STAY AT 100x, matching v3's csFillFracs convention -- both are
% coarse axes (slope up to 1.5, floor up to 0.2) where `bs150`/`bf20` read directly as 1.5/0.2.
%
% (bufferSlope, bufferFloor) = (0, 0) IS THE DEEPEST ABLATION ARM: it makes B a constant 0
% (floored to 1), so the cutoff solve returns cutoff 0 / pBoundary 0 for all three budgeted doors.
% OPTIMAL and the region-best GUARANTEE remain UNCAPPED regardless of B, so the frontier is still
% optimal + guarantee + a trickle draw, not empty.
%
csBufferSlopes = [40 80 100 120];
csBufferFloors = [5 10];
csExploreFracs = [300];
csCostFracs    = [300];
csMaxBlocks    = [4];

% The derived operating point that --single-point selects. EVERY component must be a member of its
% list, because the flag selects BY VALUE -- a derived point outside the grid would run nothing.
csDerivedBufferSlope = 100;        % bufferSlope 1.0 -> round(100 * 1.0); middle of {0,1.0,1.5}
csDerivedBufferFloor = 10;         % bufferFloor 0.1 -> round(100 * 0.1); middle of {0,0.1,0.2}
csDerivedExploreFrac = 300;        % explore_frac 0.3 -> round(1000 * 0.3); the only grid value now
csDerivedCostFrac    = 300;        % cost_frac 0.3 -> round(1000 * 0.3); the only grid value now
csDerivedMaxBlocks   = 4;

% CleanCost baseline point - one series, the well-tuned operating point. Same label format as the
% cost sweep, so its historical CSVs load here unchanged.
cleanBaseR2  = 'off';
cleanBaseW   = 90;
cleanBaseK   = 100;
cleanBaseCap = 3;

% KPAXCap cap sweep - must match KPAXCAP_CAPS in the benchmark. Values are the label tokens
% (100 x the float), exactly as they appear in the filenames.
kpaxCapCaps = [3];

% TWO REAL AXES NOW, TWO STYLE CHANNELS. v3 had three swept fractions and three channels
% (colour/style/marker); v3.2 fixes explore_frac and cost_frac, leaving only bufferSlope and
% bufferFloor to encode, so the marker channel is retired -- inventing a third visual channel for
% an axis that no longer varies would be noise, not information.
%
% colour = bufferFloor, because it is the ramp's starting value -- what fill_frac WAS, and the
% closest analogue to v3's own colour channel; DARKER IS A SMALLER STARTING BUDGET. Every B on this
% grid stays below NUM_R1_REGIONS for at least part of a run, so the ramp does not cleanly separate
% "soft" from "binding" -- it separates HOW LONG each series binds for before nActive overtakes it,
% and that window now itself grows over the run wherever bufferSlope > 0.
%   rows: floor 0 (B starts at 0, floored to 1), floor 0.1 (B0 ~ 333), floor 0.2 (B0 ~ 667)
fillColors   = [0.08 0.08 0.08;    % floor 0     smallest starting B
                0.20 0.45 0.66;    % floor 0.1
                0.55 0.68 0.84];   % floor 0.2   largest starting B
% style = bufferSlope. Solid is the bufferSlope = 0 series -- v3's constant-B design exactly -- so
% every dashed/dotted series in a figure is being read directly against its own colour's solid line.
fracStyles   = {'-', '--', ':'};   % bufferSlope = 0, 1.0, 1.5  (solid = v3's constant-B control)

% CleanCost baseline: crimson, distinct from every budget colour, drawn as a reference anchor.
cleanColor = [0.70 0.15 0.20];

% KPAXCap: grey-green, distinct from both the budget ramp and the near-black KPAX it is compared
% to. One row per entry in kpaxCapCaps; the second is kept for when cap 0.10 is restored.
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
% Each series' bufferFloor token, NaN for anything that is not a CountingStars arm. Used as the
% colour lookup and as the "is this CountingStars" NaN guard at several panels below.
%
% B ITSELF IS NOT CARRIED HERE ANY MORE. v2 had to, because B was a per-run setting and not in the
% data; v3 derives it inside the planner and logs it as the goal_frontier_size COLUMN, so the budget
% figure reads its divisor straight out of the CSV. That removes the last place the plot script had
% to know a piece of the planner's arithmetic.
plannerBufferFloor = [];

for di = 1:numel(deltas)
    dWidth = deltaWidths(di);
    dTag   = deltaTitles{di};
    dOne   = deltaSingleCap(di);   % this delta ran --single-point: only capDerived exists
    dPlus  = deltaPlusOnly(di);    % this delta ran --only-kinopaxplus: no other arm exists here

    if ~dPlus

    % --- CountingStars: bufferSlope x bufferFloor, a full factorial. explore_frac/cost_frac are
    % single-element arrays (fixed at 0.3 this pass) and maxBlocks is held at csDerivedMaxBlocks, so
    % these inner loops are trivial -- kept for structural parity with the .cu's loop nest and with
    % cross_check_countingstars_grid.py's parsing, and so re-expanding either axis later needs no
    % shape change here. ---
    for bi = 1:numel(csBufferSlopes)
        for fi = 1:numel(csBufferFloors)
            for ei = 1:numel(csExploreFracs)
                for ci = 1:numel(csCostFracs)
                    sSlope = csBufferSlopes(bi);
                    sFloor = csBufferFloors(fi);
                    eFrac  = csExploreFracs(ei);
                    cFrac  = csCostFracs(ci);
                    maxB   = csDerivedMaxBlocks;

                    % Mirror countingStarsSkip(): --single-point is the only skip.
                    if dOne && ~(sSlope == csDerivedBufferSlope && sFloor == csDerivedBufferFloor ...
                                 && eFrac == csDerivedExploreFrac && cFrac == csDerivedCostFrac)
                        continue;
                    end

                    plannerNames{end + 1}   = sprintf('CountingStars_bs%d_bf%d_ef%d_cf%d_mb%d', ...
                                                      sSlope, sFloor, eFrac, cFrac, maxB); %#ok<SAGROW>
                    plannerDisplay{end + 1} = sprintf('CS slope%g floor%g [%s]', ...
                                                      sSlope / 100, sFloor / 100, dTag); %#ok<SAGROW>
                    plannerColors(end + 1, :) = fillColors(fi, :);     %#ok<SAGROW>
                    plannerStyles{end + 1}    = fracStyles{bi};        %#ok<SAGROW>
                    plannerMarkers{end + 1}   = 'o';                   %#ok<SAGROW>
                    % The bufferSlope = 0 arm is drawn thicker at every bufferFloor: it is v3's
                    % exact constant-B design, the structural control every ramped series is read
                    % against.
                    if sSlope == 0
                        plannerWidths(end + 1) = dWidth + 0.8;         %#ok<SAGROW>
                    else
                        plannerWidths(end + 1) = dWidth;               %#ok<SAGROW>
                    end
                    plannerBaseline(end + 1) = false;                  %#ok<SAGROW>
                    plannerDeltaIdx(end + 1) = di;                     %#ok<SAGROW>
                    plannerBufferFloor(end + 1) = sFloor;              %#ok<SAGROW>
                end
            end
        end
    end

    % --- CleanCost baseline: ONE point, the reference the CountingStars grid is read against ---
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
    plannerBufferFloor(end + 1) = NaN;              %#ok<SAGROW>

    % --- KPAXCap: the control arm for the cap itself, read against the KPAX baseline below ---
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
        plannerBufferFloor(end + 1) = NaN;                %#ok<SAGROW>
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
    plannerBufferFloor = [plannerBufferFloor, NaN];                                             %#ok<AGROW>

    end   % ~dPlus

    % --- KinoPaxPlus. THE ONLY ARM THAT RUNS AT EVERY DELTA, which is the entire reason the two
    % finer deltas exist in this sweep: its advantage is a tiny frontier at a fine discretisation,
    % and a small F is exactly what CountingStars' goal_frontier_size is trying to buy directly. ---
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
        % score_floor should sit flat at EPSILON = 0.01 for KPAX and decay as 1/N_active for
        % KPAXCap and CleanCost. COUNTINGSTARS DRAWS ON NEITHER HALF OF THE LEFT PANEL: it has no
        % Syclop score, so no floor, and writes NaN. It DOES write cost_scale -- the same global
        % denominator CleanCost uses in costProbExpGlobal, and the denominator of a CountingStars
        % candidate's distance. Series lacking either column are skipped by getCol/plotMeanTime.
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

        %% ---------- FIGURE: THE REALIZED BUDGET RAMP (v3.2, NEW) ----------
        % READ THIS ONE FIRST, before anything else that depends on B. goal_frontier_size was
        % always a per-iteration column, but under v3 it was constant across a run and not worth
        % its own panel; v3.2 makes it a genuine ramp, so this is the direct visual check that the
        % realized B(itr) actually matches the intended slope*x + bufferFloor before reading any
        % panel that divides by it.
        %
        % A FLAT LINE at every bufferSlope = 0 series (solid style) is the sanity check that the
        % mechanism is wired correctly -- v3's constant B, reproduced exactly. A ramped series
        % (dashed/dotted) should rise roughly linearly from its bufferFloor's starting value toward
        % (bufferSlope + bufferFloor) * MAX_TREE_SIZE / MAX_ITER at the last iteration.
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
        title({'The realized budget ramp', 'flat = bufferSlope 0 (v3''s constant B); rising = the ramp in effect'});
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
        %   budget_used / B  > 1   OVERSHOOT -- the two uncapped doors (optimal, guarantee) already
        %                          exceeded B on their own. EXPECTED at every point on this grid,
        %                          because both are bounded by the region count and not by B: one
        %                          node per region can be a region best, one per uncovered region
        %                          can be guaranteed. B(x) = floor((slope*x+floor) *
        %                          MAX_TREE_SIZE/MAX_ITER) ranges roughly 0-8500 here (bufferFloor
        %                          0-0.2 at x=0, bufferSlope+bufferFloor up to 1.7 at x=1) against
        %                          27,000 regions -- B NOW MOVES WITHIN A RUN, not just across
        %                          series, so this ratio has two moving parts.
        %
        % SO READ THIS AS A CURVE, NOT A NUMBER. B binds EARLY in a run and then stops, at an
        % iteration that moves with the WHOLE RAMP SHAPE, not a single fill_frac any more. The
        % iteration where the curve crosses 1 IS the measurement -- early is exactly where
        % time-to-first-solution is decided. If the bufferSlope=0 curves are indistinguishable from
        % the ramped ones even early, capping the guarantee (KinoPaxPlus's hysteresis is the
        % precedent) is the next lever, not another ramp.
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
        title({'Is the budget met?', 'below 1 = a door ran dry; above 1 = the optimal door alone overran B'});
        clickableLegend();

        % --- Which door filled it. Every node came through a named door and the counts are exact,
        % so a shortfall on the left has an address here.
        %
        %   optimal_count      the top door, uncapped, first claim every iteration. It must equal
        %                      admitted_cost exactly -- pass 1 counts it, pass 2 admits it, and every
        %                      optimal candidate is admitted -- so the CSV carries a free identity
        %                      check between the two accept passes.
        %   admitted_explore   the freshness door, spending explore_frac * B.
        %   admitted_costdist  THE NEW DOOR, spending cost_frac * B on the smallest cost distances.
        %                      Pinned at 0 while cost_frac > 0 means the cutoff solve is degenerate;
        %                      sitting at cost_frac * B every iteration means it works as designed.
        %   reactivated_best   the guarantee, REALISED and counted on the device. (v2's PLANNED
        %                      count, guaranteed_react, is gone with the remainder it used to size.)
        %   reactivated_cost   v3.1's CHEAPEST reactivation arm, spending the WHOLE react_frac * B
        %                      budget on the cheapest dormant nodes. This is the arm CleanCost has
        %                      and v3 did not; it should carry essentially all of Part B's
        %                      non-guarantee volume.
        %   reactivated_count  the COMPLETENESS FLOOR alone -- ~ react_floor * dormant_count, so
        %                      ~30 nodes. It was the uniform draw through v3. LARGE HERE MEANS the
        %                      floor is doing reach work it was not sized for.
        %
        % The `reactivated` column is all THREE Part B arms, so it should equal
        % reactivated_best + reactivated_cost + reactivated_count.
        %
        % THE TWO SELECTION DOORS OVERLAP. admitted_explore and admitted_costdist are a union over
        % one candidate pool, so they do not simply add: the identity is
        % admitted == optimal_count + admitted_explore + admitted_costdist - admitted_both, and the
        % overlap has its own panel below.
        subplot(1, 2, 2); hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) getCol(t, 'optimal_count'), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
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
        end
        set(gca, 'YScale', 'log'); grid on;
        xlabel('Iteration'); ylabel('nodes');
        title({'optimal (thick solid), explore (dashed), cheapest (thin solid), guarantee (dash-dot),', ...
               'cheap-reactivation (thin dashed), completeness floor (dotted) -- the six arms'});

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

        %% ---------- FIGURE: DO THE TWO SELECTION DOORS BUY DIFFERENT THINGS ----------
        % The freshness and cheapness doors select over the SAME candidate pool on independent
        % signals, so their picks overlap and admitted_both counts the overlap exactly.
        %
        %   near 0   the two signals are independent, which is the case worth having: each door is
        %            buying nodes the other would not have.
        %   near 1   the cost door is re-admitting what freshness already took (or vice versa), so
        %            one of the two fractions is being spent twice on the same nodes and the budget
        %            it was given is going to the draw instead.
        %
        % Note it can only be read where BOTH fractions are non-zero; the ablation arms have no
        % overlap to measure and are skipped by safeRatio returning NaN on a zero denominator.
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Door Overlap (%s)', envTitle, costTitle), ...
               'Position', [160 160 900 560]);
        hold on;
        for pi = 1:nPlanner
            if isnan(plannerBufferFloor(pi)), continue; end   % not a CountingStars series
            plotMeanIter(R{pi}, @(t) safeRatio(getCol(t, 'admitted_both'), ...
                                               sumCols(t, 'admitted_explore', 'admitted_costdist')), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        grid on; ylim([0 1]);
        xlabel('Iteration'); ylabel('admitted\_both / (explore + costdist)');
        title({'Selection-door overlap', ...
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

        sgtitle(sprintf('Cost-Prune Tuning Grid at %s \x2014 %s, %s (run means)', ...
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
        % \x25cb/\x25a1/\x25b3 are cost_frac 0 / 0.2 / 0.4 -- the marker channel has encoded
        % cost_frac since v3, NOT maxBlocks, which is held at 4 and is not an axis.
        % sprintf, NOT a bare concatenation: the \x.... marker glyphs and the \\_ TeX underscore
        % escapes are only resolved by a formatting call, and this string is substituted into the
        % titles below via %s -- which inserts it verbatim rather than re-interpreting it. Built as
        % a plain [...] it would print the escape sequences literally.
        markerKey = sprintf(['lower-left is better (fast and cheap); darker = smaller fill\\_frac; ' ...
                             '\x25cb/\x25a1/\x25b3 CountingStars (cost\\_frac 0/0.2/0.4), ' ...
                             '\x2606 CleanCost, \x25bd KPAXCap, \x25a1 KPAX, \x25c7 KinoPaxPlus']);

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
        % figures). CleanCost's advantage is expected to be the second, since its Part B reactivation
        % is cost-weighted and refinement is what Part B does.
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
        % (each delta contributes its own baselines and KPAXCap pair).
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
