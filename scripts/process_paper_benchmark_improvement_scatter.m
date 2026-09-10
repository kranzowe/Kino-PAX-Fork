%% Paper Benchmark Improvement Scatter - time-to-first-solution vs %% improvement over KPAX
% Reads per-iteration CSVs produced by examples/gpu/paper_benchmark.cu (run via
% scripts/run_paper_benchmark.sh) -- SEPARATE from process_paper_benchmark_and_plot.m, which
% plots one environment at a time. This script reads every configured environment subfolder (see
% `environments` below -- THREE this pass, `empty` pulled out as a diagnostic) in one pass and
% produces:
%
%   - one scatter per (environment, cost metric): mean time-to-first-solution (x) vs mean %
%     improvement in FINAL cost over KPAX, AT THE SAME DELTA (y), one point per (planner, delta).
%   - one more scatter per cost metric: the SAME thing, averaged across all configured environments
%     (the per-environment means computed above, averaged again over environment -- not a pool of
%     every run from every environment, so a low-success environment does not get implicitly
%     downweighted just for having fewer successful runs).
%
% % IMPROVEMENT is computed against KPAX'S OWN mean final cost AT THAT SAME DELTA, in that same
% environment (and, for the per-environment figures, that same cost metric) -- not against a
% single global KPAX number -- because KPAX's own performance already varies by discretization,
% so comparing every delta against one KPAX baseline would conflate "this planner is better" with
% "this delta made KPAX worse." KPAX itself is plotted too (at 0% by construction, per delta) as a
% visual anchor alongside the y=0 reference line.
%
% loadRuns() and the aggregate helpers (firstSolTime/finalCost/sanitize_name/clickableLegend) are
% verbatim copies of process_paper_benchmark_and_plot.m's -- keep both in step by hand if either
% changes; there is no shared file to import from (every plot script in this repo is self-contained
% on purpose, see e.g. process_countingstars_and_plot.m's own copies).
%
% USAGE: cd into the PARENT of the environment folders, then call BY NAME, not via run():
%   cd build/Data/Benchmarks/Paper
%   addpath('<repo>/scripts')
%   process_paper_benchmark_improvement_scatter
% run('<abs path>/...') would cd to the scripts folder first, and baseDir below ('' = current
% folder) would then find nothing.

clear; clc; close all;

%% --- Configuration ---
baseDir = '';   % '' = current directory (run this from Data/Benchmarks/Paper, the PARENT of the
                % environment subfolders -- NOT from inside one of them, unlike
                % process_paper_benchmark_and_plot.m)

% THREE environments this pass, one subfolder each under baseDir. `empty` IS NOT PART OF THIS
% PASS -- see run_paper_benchmark.sh's ENV_NAMES: a run at tiny/empty froze, and empty was pulled
% out to isolate whether it's specifically responsible. Restore the commented-out line below (and
% remove this one) once that's answered -- there will be no empty/ subfolder to load until then.
environments = {'house', 'narrowPassage', 'zigzag'};
envTitles    = {'House', 'Narrow Passage', 'Zigzag Corridor (tightened)'};
% environments = {'empty', 'house', 'narrowPassage', 'zigzag'};
% envTitles    = {'Empty', 'House', 'Narrow Passage', 'Zigzag Corridor (tightened)'};

% Cost metric axis -- must match run_paper_benchmark.sh's COST_LABELS.
metrics       = {'length', 'effort'};
metricTitles  = {'Workspace Path Length', 'Control Effort'};

% Delta axis. Must match run_paper_benchmark.sh's DELTA_LABELS. Encoded here as SCATTER MARKER
% SIZE (small -> large for large -> fine -> tiny), since colour/marker shape are spent on planner
% identity below and this is a scatter, not a line plot (no style/width channel to give delta).
deltas          = {'large', 'fine', 'tiny'};
deltaTitles     = {'9k', '262k W-refined', '593k V-refined'};
deltaMarkerSizes = [7, 10, 13];

% --- The four FIXED series (not swept). Label tokens must match examples/gpu/paper_benchmark.cu's
% trueLabel() / countingStarsLabel() exactly -- copied verbatim from
% process_paper_benchmark_and_plot.m; keep both in step by hand if either changes. ---
trueCap           = 100;   % syclopCap 1.0 (no cap)
trueAncestorPrune = 1;     % the guarded-prune point; anc0 (pure fusion) was dropped from this pass
csSlope = 120; csFloor = 40; csExplore = 150; csCost = 750; csHopelessGuard = 1;

baseNames = { ...
    'KPAX', ...
    'KinoPaxPlus', ...
    sprintf('KinoPaxSTARTrue_cap%d_anc%d', trueCap, trueAncestorPrune), ...
    sprintf('CountingStars_bs%d_bf%d_ef%d_cf%d_hg%d', csSlope, csFloor, csExplore, csCost, csHopelessGuard) ...
};
baseDisplay = { ...
    'KPAX', ...
    'KinoPaxPlus', ...
    'KinoPaxSTARTrue (naive, stale-best prune)', ...
    'CountingStars (slope 1.2, floor 0.4, ef 0.15, cf 0.75, hopeless guard ON)' ...
};
% color = planner identity, same family as process_paper_benchmark_and_plot.m's baseColors.
baseColors = [ ...
    0.10 0.10 0.10;    % KPAX
    0.20 0.40 0.80;    % KinoPaxPlus
    0.45 0.05 0.10;    % KinoPaxSTARTrue (crimson)
    0.85 0.55 0.10 ];  % CountingStars (amber)
baseMarkers = {'s', 'd', 'h', 'o'};
kpaxBaseIdx = find(strcmp(baseNames, 'KPAX'), 1);
if isempty(kpaxBaseIdx)
    error('KPAX not found in baseNames -- it is the %% improvement reference and must be present.');
end

% --- Build the (planner, delta) series arrays -- planner-major, same convention as
% process_paper_benchmark_and_plot.m, so plannerDisplay/plannerBaseIdx line up the same way. ---
plannerNames   = {};
plannerDisplay = {};
plannerColors  = [];
plannerMarkers = {};
plannerSizes   = [];
plannerDeltaIdx = [];   % index into `deltas`
plannerBaseIdx  = [];   % index into `baseNames`/`baseDisplay`

for si = 1:numel(baseNames)
    for di = 1:numel(deltas)
        plannerNames{end + 1}    = baseNames{si};                                %#ok<SAGROW>
        plannerDisplay{end + 1}  = sprintf('%s [%s]', baseDisplay{si}, deltaTitles{di}); %#ok<SAGROW>
        plannerColors(end + 1, :) = baseColors(si, :);                           %#ok<SAGROW>
        plannerMarkers{end + 1}  = baseMarkers{si};                              %#ok<SAGROW>
        plannerSizes(end + 1)    = deltaMarkerSizes(di);                         %#ok<SAGROW>
        plannerDeltaIdx(end + 1) = di;                                           %#ok<SAGROW>
        plannerBaseIdx(end + 1)  = si;                                           %#ok<SAGROW>
    end
end

numRunsPer = 20;             % max runs searched per series (missing files skipped); harness writes 5
MAX_FLOAT_THRESH = 1e30;     % best_cost sentinel (MAX_FLOAT / INFINITY) -> NaN
nPlanner = numel(plannerNames);

%% ======================================================================
figNum = 0;
for mi = 1:numel(metrics)
    metric    = metrics{mi};
    costTitle = metricTitles{mi};
    fprintf('\n=== Cost metric: %s ===\n', costTitle);

    % Per-environment means, accumulated across the environment loop so they can be averaged
    % afterward -- ei rows, one (planner,delta) column each, matching plannerNames' order.
    allFirstSolTime = NaN(numel(environments), nPlanner);
    allFinalCost    = NaN(numel(environments), nPlanner);
    allPctImprove   = NaN(numel(environments), nPlanner);

    for ei = 1:numel(environments)
        env      = environments{ei};
        envTitle = envTitles{ei};
        envDir   = fullfile(baseDir, env);
        fprintf('\n--- Environment: %s ---\n', env);

        mFirstSolTime = NaN(1, nPlanner);
        mFinalCost    = NaN(1, nPlanner);
        for pi = 1:nPlanner
            tok  = sprintf('%s_%s', deltas{plannerDeltaIdx(pi)}, metric);
            runs = loadRuns(envDir, env, plannerNames{pi}, tok, numRunsPer);
            fprintf('  %-34s %-18s : %d runs\n', plannerNames{pi}, ...
                    ['[' deltas{plannerDeltaIdx(pi)} ']'], numel(runs));

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

        % --- % improvement over KPAX, matched delta-for-delta within this environment. ---
        mPctImprove = NaN(1, nPlanner);
        for di = 1:numel(deltas)
            kIdx = find(plannerBaseIdx == kpaxBaseIdx & plannerDeltaIdx == di, 1);
            kpaxCost = mFinalCost(kIdx);
            if isnan(kpaxCost) || kpaxCost <= 0, continue; end
            for pi = find(plannerDeltaIdx == di)
                if isnan(mFinalCost(pi)), continue; end
                mPctImprove(pi) = 100 * (kpaxCost - mFinalCost(pi)) / kpaxCost;
            end
        end

        allFirstSolTime(ei, :) = mFirstSolTime;
        allFinalCost(ei, :)    = mFinalCost;
        allPctImprove(ei, :)   = mPctImprove;

        figNum = figNum + 1;
        figure('Name', sprintf('%s - Improvement Scatter (%s)', envTitle, costTitle), ...
               'Position', [120 + 20 * ei, 120 + 15 * ei, 1180, 700]);
        improvementScatter(mFirstSolTime, mPctImprove, plannerMarkers, plannerColors, ...
                           plannerSizes, plannerDisplay);
        xlabel('Avg Time to First Solution (ms)');
        ylabel(sprintf('%% Improvement over KPAX (Final %s, same delta)', costTitle));
        title(sprintf(['Time to First Solution vs %% Improvement over KPAX \x2014 %s, %s\n' ...
                       'marker size = delta (small->large = large->fine->tiny); KPAX itself sits at 0%%'], ...
                       envTitle, costTitle), 'FontWeight', 'bold');

        writeImprovementTable(plannerDisplay, deltas, plannerDeltaIdx, mFirstSolTime, mFinalCost, ...
                              mPctImprove, envDir, sprintf('paper_benchmark_improvement_%s_%s.csv', ...
                              sanitize_name(env), metric));
    end

    % --- Average across environments: mean of the per-environment MEANS, not a pooled re-mean
    % over every run -- an environment with a low success rate does not get implicitly
    % downweighted just for having fewer successful runs contributing to its own mean. ---
    avgFirstSolTime = mean(allFirstSolTime, 1, 'omitnan');
    avgFinalCost    = mean(allFinalCost, 1, 'omitnan');
    avgPctImprove   = mean(allPctImprove, 1, 'omitnan');

    figNum = figNum + 1;
    figure('Name', sprintf('Average - Improvement Scatter (%s)', costTitle), ...
           'Position', [160, 160, 1180, 700]);
    improvementScatter(avgFirstSolTime, avgPctImprove, plannerMarkers, plannerColors, ...
                       plannerSizes, plannerDisplay);
    xlabel('Avg Time to First Solution (ms)');
    ylabel(sprintf('%% Improvement over KPAX (Final %s, same delta)', costTitle));
    title(sprintf(['Time to First Solution vs %% Improvement over KPAX \x2014 Average Across ' ...
                   'Environments, %s\nmarker size = delta (small->large = large->fine->tiny); ' ...
                   'averaged over environment means, not pooled runs'], costTitle), ...
          'FontWeight', 'bold');

    writeImprovementTable(plannerDisplay, deltas, plannerDeltaIdx, avgFirstSolTime, avgFinalCost, ...
                          avgPctImprove, baseDir, sprintf('paper_benchmark_improvement_average_%s.csv', metric));
end

fprintf('\nGenerated %d figures.\n', figNum);

%% ======================================================================
%% Helper functions (local functions)
%% ======================================================================

function improvementScatter(x, y, markers, colors, sizes, labels)
    % One point per (planner, delta); colour+marker shape = planner identity, marker SIZE = delta.
    % A y=0 reference line is the "no improvement over KPAX" anchor -- KPAX's own points sit on it
    % by construction.
    hold on;
    for pi = 1:numel(x)
        if isnan(x(pi)) || isnan(y(pi)), continue; end
        plot(x(pi), y(pi), markers{pi}, ...
             'MarkerFaceColor', colors(pi, :), ...
             'MarkerEdgeColor', 'k', 'LineWidth', 0.5, ...
             'MarkerSize', sizes(pi), 'DisplayName', labels{pi});
    end
    yline(0, 'k--', 'no improvement over KPAX', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    grid on;
    clickableLegend();
end

function writeImprovementTable(plannerDisplay, deltas, plannerDeltaIdx, firstSolTimeVals, ...
                               finalCostVals, pctImproveVals, outDir, csvName)
    nPlanner = numel(plannerDisplay);
    tPlanner = cell(nPlanner, 1);
    tDelta   = cell(nPlanner, 1);
    for pi = 1:nPlanner
        tPlanner{pi} = plannerDisplay{pi};
        tDelta{pi}   = deltas{plannerDeltaIdx(pi)};
    end
    fprintf('%-55s %-18s %16s %14s %16s\n', 'Planner', 'Delta', 'TimeToFirst(ms)', 'FinalCost', 'PctImproveKPAX');
    for pi = 1:nPlanner
        fprintf('%-55s %-18s %16.2f %14.4f %16.2f\n', tPlanner{pi}, tDelta{pi}, ...
                firstSolTimeVals(pi), finalCostVals(pi), pctImproveVals(pi));
    end
    T = table(tPlanner, tDelta, firstSolTimeVals(:), finalCostVals(:), pctImproveVals(:), ...
              'VariableNames', {'Planner', 'Delta', 'TimeToFirst_ms', 'FinalCost', 'PctImprovementOverKPAX'});
    writetable(T, fullfile(outDir, csvName));
    fprintf('Table written to: %s\n', csvName);
end

function runs = loadRuns(dataDir, env, planner, delta, numRuns)
    % Load one (planner, delta) series' per-run CSVs; missing files are skipped. Verbatim copy of
    % process_paper_benchmark_and_plot.m's loadRuns() -- keep both in step by hand.
    runs = {};
    for ri = 0:(numRuns - 1)
        switch planner
            case 'KinoPaxPlus'
                fn = sprintf('%s_delta%s_run%d.csv', env, delta, ri);
            case 'KPAX'
                fn = sprintf('%s_KPAX_delta%s_run%d.csv', env, delta, ri);
            otherwise
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

function s = sanitize_name(s)
    % Filesystem-safe token for a CSV filename. Verbatim copy of
    % process_paper_benchmark_and_plot.m's version.
    s = regexprep(s, '[^a-zA-Z0-9_-]', '_');
end

function t = firstSolTime(tbl, thresh)
    % Elapsed time (ms) at which this run first reached a finite (< thresh) best_cost;
    % -1 if it never found a solution. Verbatim copy of process_paper_benchmark_and_plot.m's.
    solIdx = find(tbl.best_cost < thresh, 1, 'first');
    if isempty(solIdx), t = -1; else, t = tbl.elapsed_time_ms(solIdx); end
end

function c = finalCost(tbl, thresh)
    % Verbatim copy of process_paper_benchmark_and_plot.m's finalCost().
    costs = tbl.best_cost;
    costs(costs > thresh) = NaN;
    v = costs(~isnan(costs));
    if isempty(v), c = NaN; else, c = v(end); end
end
