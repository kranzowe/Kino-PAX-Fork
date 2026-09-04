%% CountingStars Summary Plots - zigzag, length + effort, 3 panels per metric
% Reads per-iteration CSVs produced by examples/gpu/countingstars_sweep.cu
% (run via scripts/run_countingstars_sweep.sh).
%
% A DELIBERATELY NARROWED VIEW of process_countingstars_and_plot.m, which carries every diagnostic
% panel (budget invariants, door overlap, selection cutoffs, fan-out, ...). This script draws only
% the three panels that answer the headline question -- does a variant win, and how -- one figure
% each, once per cost metric:
%
%   1. Best Cost vs Time        the fair cross-planner axis (an "iteration" is a different unit of
%                                work per planner; elapsed time is not).
%   2. Tradeoff Scatter         mean time-to-first-solution vs mean FINAL cost, one point per
%                                variant. Lower-left wins both.
%   3. Tree Growth vs Iteration how fast the tree actually fills, against nothing (there is no
%                                growth controller here -- growth is an OUTPUT of the doors, not a
%                                target).
%
% For every other panel (budget ramp, door counts, selection cutoffs, fan-out budget, block
% identities, ...), run process_countingstars_and_plot.m instead -- it reads the same CSVs, this
% script's series/loading code is a direct copy of its relevant parts, and nothing here changes
% what got written to disk.
%
% 2 metrics x 3 panels = 6 figures.
%
% Series are (planner, delta) pairs, overlaid inside each figure (delta encoded as line width) --
% see run_countingstars_sweep.sh's DELTA_LABELS / DELTA_EXTRA_ARGS. THIS SCRIPT'S GRID MIRROR
% (csBufferSlopes etc. below) MUST STAY IN STEP WITH BUFFER_SLOPES/BUFFER_FLOORS/EXPLORE_FRACS/
% COST_FRACS in countingstars_sweep.cu -- cross_check_countingstars_grid.py checks the two against
% process_countingstars_and_plot.m, not against this file, so a drift here fails silently: loadRuns
% reports "0 runs" for the orphaned series instead of erroring. Keep this file's grid arrays
% identical to process_countingstars_and_plot.m's if you change the sweep.
%
% USAGE: cd into the data directory, then call the script BY NAME, not via run():
%   cd build/Data/Benchmarks/CountingStars/zigzag
%   addpath('<repo>/scripts')
%   process_countingstars_summary_plots
% run('<abs path>/process_countingstars_summary_plots.m') would cd to the scripts folder first,
% and dataDir below ('' = current folder) would then find nothing.

clear; clc; close all;

%% --- Configuration ---
dataDir = '';   % '' = current directory (run this from Data/Benchmarks/CountingStars/<env>)

% One environment per run — must match the subfolder you cd'd into.
environments = {'zigzag'};
envTitles    = {'Zigzag Corridor'};

% Cost metric axis — one build each, so one set of figures each. BOTH this pass.
metrics       = {'length', 'effort'};
metricTitles  = {'Workspace Path Length', 'Control Effort'};
metricYLabels = {'Path Cost (workspace path length)', 'Path Cost (control effort)'};

% Delta axis — OVERLAID inside each figure, encoded as line WIDTH. The filename token is
% sprintf('%s_%s', delta, metric), e.g. 'fine_effort'.
deltas      = {'large', 'fine', 'tiny'};
deltaTitles = {'27k', '216k W-refined', '593k V-refined'};
deltaWidths = [1.0, 1.8, 2.6];

% WHICH ARMS EXIST AT EACH DELTA. Index 0 runs the full sweep; the two finer deltas run
% KINOPAXPLUS ONLY -- must match DELTA_EXTRA_ARGS in run_countingstars_sweep.sh.
deltaPlusOnly = [false, true, true];

capDerived     = 3;       % label token for cap = 0.03 (CAP_DERIVED in the benchmark)
deltaSingleCap = [false, false, false];   % --single-point not used by this sweep

deltaLabel = '3 deltas overlaid';

% CountingStars grid -- MUST match BUFFER_SLOPES / BUFFER_FLOORS / EXPLORE_FRACS / COST_FRACS in
% countingstars_sweep.cu. Label tokens: bufferSlope/bufferFloor as round(100 x float), the two
% shares as round(1000 x float).
csBufferSlopes = [140 180 220];
csBufferFloors = [5 20];
csExploreFracs = [300];
csCostFracs    = [300];

% The derived operating point that --single-point selects (unused this pass, kept for parity).
csDerivedBufferSlope = 180;
csDerivedBufferFloor = 5;
csDerivedExploreFrac = 300;
csDerivedCostFrac    = 300;

% CleanCost baseline point - one series, the well-tuned operating point.
cleanBaseR2  = 'off';
cleanBaseW   = 90;
cleanBaseK   = 100;
cleanBaseCap = 3;

% KPAXCap cap sweep - must match KPAXCAP_CAPS in the benchmark.
kpaxCapCaps = [3];

% colour = bufferFloor (darker = smaller starting budget); style = bufferSlope; marker =
% (explore_frac, cost_frac) pair, inert this pass (both single-element, fixed at 0.3).
fillColors  = [0.08 0.08 0.08;    % floor 0.05   smallest starting B
               0.55 0.68 0.84];   % floor 0.2    largest starting B
fracStyles  = {'-', '--', ':'};   % bufferSlope = 1.4, 1.8, 2.2 (in csBufferSlopes order)
efCfMarkers = {'o', '^', '>', '<'};   % one per (explore_frac, cost_frac) combination -- only 'o'
                                        % is ever used while both axes are single-element

cleanColor = [0.70 0.15 0.20];    % CleanCost: crimson reference anchor
mossRamp   = [0.58 0.73 0.53;     % KPAXCap cap 0.03 (lighter)
              0.24 0.44 0.26];    % KPAXCap cap 0.10 (darker; unused unless kpaxCapCaps grows)

% --- Build the series arrays: (planner, delta) pairs ---
plannerNames    = {};
plannerDisplay  = {};
plannerColors   = [];
plannerStyles   = {};
plannerMarkers  = {};
plannerWidths   = [];
plannerBaseline = [];   % logical: drawn as a large scatter marker
plannerDeltaIdx = [];   % index into `deltas`

for di = 1:numel(deltas)
    dWidth = deltaWidths(di);
    dTag   = deltaTitles{di};
    dOne   = deltaSingleCap(di);
    dPlus  = deltaPlusOnly(di);

    if ~dPlus

    numEfCf = numel(csExploreFracs) * numel(csCostFracs);
    assert(numEfCf <= numel(efCfMarkers), ...
        sprintf(['efCfMarkers has %d entries but csExploreFracs x csCostFracs needs %d -- add more ' ...
                 'marker shapes to efCfMarkers before growing either axis.'], numel(efCfMarkers), numEfCf));
    for bi = 1:numel(csBufferSlopes)
        for fi = 1:numel(csBufferFloors)
            for ei = 1:numel(csExploreFracs)
                for ci = 1:numel(csCostFracs)
                    sSlope = csBufferSlopes(bi);
                    sFloor = csBufferFloors(fi);
                    eFrac  = csExploreFracs(ei);
                    cFrac  = csCostFracs(ci);
                    efCfIdx = (ei - 1) * numel(csCostFracs) + ci;

                    if dOne && ~(sSlope == csDerivedBufferSlope && sFloor == csDerivedBufferFloor ...
                                 && eFrac == csDerivedExploreFrac && cFrac == csDerivedCostFrac)
                        continue;
                    end

                    plannerNames{end + 1}   = sprintf('CountingStars_bs%d_bf%d_ef%d_cf%d', ...
                                                      sSlope, sFloor, eFrac, cFrac); %#ok<SAGROW>
                    plannerDisplay{end + 1} = sprintf('CS slope%g floor%g ef%g cf%g [%s]', ...
                                                      sSlope / 100, sFloor / 100, ...
                                                      eFrac / 1000, cFrac / 1000, dTag); %#ok<SAGROW>
                    plannerColors(end + 1, :) = fillColors(fi, :);     %#ok<SAGROW>
                    plannerStyles{end + 1}    = fracStyles{bi};        %#ok<SAGROW>
                    plannerMarkers{end + 1}   = efCfMarkers{efCfIdx};  %#ok<SAGROW>
                    if sSlope == min(csBufferSlopes)
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

    % --- CleanCost baseline ---
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

    % --- KinoPaxPlus. Runs at every delta. ---
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
        R = cell(1, nPlanner);
        for pi = 1:nPlanner
            tok   = sprintf('%s_%s', deltas{plannerDeltaIdx(pi)}, metric);
            R{pi} = loadRuns(dataDir, env, plannerNames{pi}, tok, numRunsPer(pi));
            fprintf('  %-34s %-18s : %d runs\n', plannerNames{pi}, ...
                    ['[' deltas{plannerDeltaIdx(pi)} ']'], numel(R{pi}));
        end

        %% ---------- FIGURE 1: Best Cost vs Time (mean lines, no bands) ----------
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

        %% ---------- FIGURE 2: Tree Growth vs Iteration ----------
        % THERE IS NO GROWTH CONTROLLER -- this is an OUTPUT of however many candidates the doors
        % admitted, not a target the planner tracks against a reference line.
        figNum = figNum + 1;
        figure('Name', sprintf('%s - Tree Growth (%s)', envTitle, costTitle), ...
               'Position', [70 70 1000 640]);
        hold on;
        for pi = 1:nPlanner
            plotMeanIter(R{pi}, @(t) getCol(t, 'tree_size'), ...
                         plannerColors(pi, :), plannerStyles{pi}, plannerWidths(pi), plannerDisplay{pi});
        end
        xlabel('Iteration'); ylabel('tree\_size'); grid on;
        clickableLegend();
        title(sprintf('Tree Growth vs Iteration \x2014 %s, %s, %s', envTitle, deltaLabel, costTitle), ...
              'FontWeight', 'bold');

        %% ---------- Aggregate summary metrics per planner (for FIGURE 3) ----------
        mFirstSolTime = NaN(1, nPlanner);
        mFinalCost    = NaN(1, nPlanner);
        for pi = 1:nPlanner
            runs = R{pi};
            fstVals = []; fcVals = [];
            for ri = 1:numel(runs)
                if isempty(runs{ri}), continue; end
                ft = firstSolTime(runs{ri}, MAX_FLOAT_THRESH);
                if ft >= 0, fstVals(end + 1) = ft; end %#ok<SAGROW>
                fc = finalCost(runs{ri}, MAX_FLOAT_THRESH);
                if ~isnan(fc), fcVals(end + 1) = fc; end %#ok<SAGROW>
            end
            if ~isempty(fstVals), mFirstSolTime(pi) = mean(fstVals); end
            if ~isempty(fcVals),  mFinalCost(pi)    = mean(fcVals);  end
        end

        %% ---------- FIGURE 3: Tradeoff Scatter, Time to First Solution vs Final Cost ----------
        costLims = mFinalCost(isfinite(mFinalCost));
        if numel(costLims) >= 2 && max(costLims) > min(costLims)
            pad      = 0.05 * (max(costLims) - min(costLims));
            costYLim = [min(costLims) - pad, max(costLims) + pad];
        else
            costYLim = [];   % nothing solved, or a single value: let MATLAB autoscale
        end

        markerKey = sprintf(['lower-left is better (fast and cheap); darker = smaller bufferFloor; ' ...
                             '\x25cb CountingStars, ' ...
                             '\x2606 CleanCost, \x25bd KPAXCap, \x25a1 KPAX, \x25c7 KinoPaxPlus']);

        figNum = figNum + 1;
        figure('Name', sprintf('%s - Tradeoff Scatter (%s)', envTitle, costTitle), ...
               'Position', [130 140 1180 700]);
        tradeoffScatter(mFirstSolTime, mFinalCost, plannerMarkers, plannerColors, ...
                        plannerBaseline, plannerDisplay, costYLim);
        xlabel('Avg Time to First Solution (ms)'); ylabel(sprintf('Avg Final %s', costYLab));
        title(sprintf(['Tuning Tradeoff: Time to First Solution vs FINAL Cost \x2014 %s, %s\n%s'], ...
                       envTitle, costTitle, markerKey), 'FontWeight', 'bold');
    end
end

fprintf('\nGenerated %d figures.\n', figNum);

%% ======================================================================
%% Helper functions (local functions; copies of process_countingstars_and_plot.m's, trimmed to
%% only what these three panels need)
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
    % longest-lasting run's).
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
    A  = sanitize(A);
    mu = mean(A, 1, 'omitnan');
    valid = ~isnan(mu);
    if ~any(valid), return; end
    plot(commonTime(valid), mu(valid), style, 'Color', color, ...
         'LineWidth', width, 'DisplayName', name);
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

function clickableLegend()
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
    A(A > 1e30) = NaN;
    A(A == -1)  = NaN;
end

function t = firstSolTime(tbl, thresh)
    solIdx = find(tbl.best_cost < thresh, 1, 'first');
    if isempty(solIdx), t = -1; else, t = tbl.elapsed_time_ms(solIdx); end
end

function c = finalCost(tbl, thresh)
    costs = tbl.best_cost;
    costs(costs > thresh) = NaN;
    v = costs(~isnan(costs));
    if isempty(v), c = NaN; else, c = v(end); end
end

function tradeoffScatter(x, y, markers, colors, isBaseline, labels, yLim)
    hold on;
    for pi = 1:numel(x)
        if isnan(x(pi)) || isnan(y(pi)), continue; end
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
