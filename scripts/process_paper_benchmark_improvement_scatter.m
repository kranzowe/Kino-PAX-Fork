%% Paper Benchmark Improvement Scatter - time-to-first-solution vs % improvement over KPAX
% Reads per-iteration CSVs produced by examples/gpu/paper_benchmark_v2.cu (run via
% scripts/run_paper_benchmark_v2.sh) -- SEPARATE from process_paper_benchmark_v2_and_plot.m, which
% plots one vehicle model's data at a time. This script reads ALL FOUR environments AND ALL THREE
% vehicle models in one pass (see `environments`/`models` below) and produces THREE LEVELS of
% aggregation:
%
%   1. PER ENVIRONMENT (one scatter per (environment, cost metric)): mean time-to-first-solution
%      (x) vs mean % improvement in FINAL cost over KPAX, AT THE SAME MODEL (y) -- one point per
%      (planner, model), marker SIZE encodes model.
%   2. PER MODEL, averaged across environments -- the "intermediate" averages (one scatter per
%      (model, cost metric)): each (planner, model) point from level 1 is averaged across all 4
%      environments -- one point per planner, for that one model.
%   3. GRAND AVERAGE, across ALL environments AND ALL models (one scatter per cost metric): each
%      planner's level-2 per-model averages are averaged again, across all 3 models -- one point
%      per planner, model fully collapsed away.
%
% Both averaging steps are MEAN OF MEANS, never a pooled re-mean over raw runs -- an
% environment/model with a lower success rate does not get implicitly downweighted just for having
% fewer successful runs contributing to its own mean (same principle applied at both levels: level
% 3 averages level 2's per-model numbers, not the raw (environment x model) grid).
%
% % IMPROVEMENT is computed against KPAX'S OWN mean final cost AT THAT SAME MODEL, in that same
% environment (and, for the per-environment figures, that same cost metric) -- not against a single
% global KPAX number -- because KPAX's own performance already varies by vehicle model, so
% comparing every model against one KPAX baseline would conflate "this planner is better" with
% "this model made KPAX worse." KPAX itself is plotted too (at 0% by construction) as a visual
% anchor alongside the y=0 reference line.
%
% GRAND-AVERAGE FinalCost IS DELIBERATELY OMITTED (kept at the per-environment and per-model levels
% only, written as NaN in the grand table/CSV): raw cost is only comparable WITHIN one vehicle
% model -- Quad's workspace is [0,100]^3 while Double Integrator/Dubins Airplane are [0,1]^3, so a
% "final cost" blended across all three models would mix physically different units into one
% meaningless number. Time-to-first-solution (ms) and % improvement (already normalized against
% each model's own KPAX) are both still valid to blend across models, so both remain at every level.
%
% Each vehicle model now runs at exactly ONE discretization (no more large/fine/tiny sweep -- see
% run_paper_benchmark_v2.sh), so MODEL is the axis this script sweeps/encodes instead of delta. The
% on-disk delta TOKEN embeds the model (e.g. "m1_tiny_length", "m2_tiny_effort", "m3_tiny_length"
% -- see run_paper_benchmark_v2.sh's TAG) so that all three models' CSVs can coexist in the same
% per-environment folder without colliding.
%
% loadRuns() and the aggregate helpers (firstSolTime/finalCost/sanitize_name/clickableLegend) are
% verbatim copies of process_paper_benchmark_v2_and_plot.m's -- keep both in step by hand if either
% changes; there is no shared file to import from (every plot script in this repo is self-contained
% on purpose, see e.g. process_countingstars_and_plot.m's own copies).
%
% USAGE: cd into the PARENT of the environment folders, then call BY NAME, not via run():
%   cd build/Data/Benchmarks/PaperBenchmarkV2
%   addpath('<repo>/scripts')
%   process_paper_benchmark_improvement_scatter
% run('<abs path>/...') would cd to the scripts folder first, and baseDir below ('' = current
% folder) would then find nothing.

clear; clc; close all;

%% --- Configuration ---
baseDir = '';   % '' = current directory (run this from Data/Benchmarks/PaperBenchmarkV2, the
                % PARENT of the environment subfolders -- NOT from inside one of them, unlike
                % process_paper_benchmark_v2_and_plot.m)

% ALL FOUR ENVIRONMENTS -- paper_benchmark_v2.cu runs on countingstars_sweep.cu's harness (unlike
% the old paper_benchmark.cu, which hung at tiny/empty and needed `empty` pulled out as a
% diagnostic), so `empty` is included this pass same as the other three.
environments = {'empty', 'house', 'narrowPassage', 'zigzag'};
envTitles    = {'Empty', 'House', 'Narrow Passage', 'Zigzag Corridor (tightened)'};

% Cost metric axis -- must match run_paper_benchmark_v2.sh's COST_LABELS.
metrics       = {'length', 'effort'};
metricTitles  = {'Workspace Path Length', 'Control Effort'};

% Model axis -- must match run_paper_benchmark_v2.sh's MODEL_IDS. Encoded here as SCATTER MARKER
% SIZE (small -> large), since colour/marker shape are spent on planner identity below and this is
% a scatter, not a line plot. Each model runs at exactly ONE discretization now (modelDeltaLabel),
% so there is no separate delta axis left to sweep -- see run_paper_benchmark_v2.sh's MODEL_W_R1S/
% MODEL_C_R1S/MODEL_V_R1S for each model's actual region count.
models           = [1, 2, 3];
modelNames       = {'DoubleIntegrator', 'DubinsAirplane', 'Quad'};   % short, filename-safe tokens
modelTitles      = {'6D Double Integrator', '6D Dubins Airplane', '12D Non-Linear Quad'};  % display
modelDeltaLabel  = 'tiny';    % all three models currently share this one delta label
modelMarkerSizes = [7, 10, 13];
GRAND_MARKER_SIZE = 12;   % no model axis left to encode once models are averaged away (level 3)

% --- The four FIXED series (not swept). Label tokens must match examples/gpu/paper_benchmark_v2.cu's
% trueLabel() / countingStarsLabel() exactly -- copied verbatim from
% process_paper_benchmark_v2_and_plot.m; keep both in step by hand if either changes. ---
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
% color = planner identity, same family as process_paper_benchmark_v2_and_plot.m's baseColors.
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

% --- Build the (planner, model) series arrays -- planner-major, same convention as
% process_paper_benchmark_v2_and_plot.m, so plannerDisplay/plannerBaseIdx line up the same way. ---
plannerNames    = {};
plannerDisplay  = {};
plannerColors   = [];
plannerMarkers  = {};
plannerSizes    = [];
plannerModelIdx = [];   % index into `models`
plannerBaseIdx  = [];   % index into `baseNames`/`baseDisplay`

for si = 1:numel(baseNames)
    for mo = 1:numel(models)
        plannerNames{end + 1}    = baseNames{si};                                      %#ok<SAGROW>
        plannerDisplay{end + 1}  = sprintf('%s [%s]', baseDisplay{si}, modelNames{mo}); %#ok<SAGROW>
        plannerColors(end + 1, :) = baseColors(si, :);                                 %#ok<SAGROW>
        plannerMarkers{end + 1}  = baseMarkers{si};                                    %#ok<SAGROW>
        plannerSizes(end + 1)    = modelMarkerSizes(mo);                               %#ok<SAGROW>
        plannerModelIdx(end + 1) = mo;                                                 %#ok<SAGROW>
        plannerBaseIdx(end + 1)  = si;                                                 %#ok<SAGROW>
    end
end

numRunsPer = 35;             % max runs searched per series (missing files skipped); harness writes 30
MAX_FLOAT_THRESH = 1e30;     % best_cost sentinel (MAX_FLOAT / INFINITY) -> NaN
nPlanner = numel(plannerNames);   % 4 baseNames x 3 models = 12

%% ======================================================================
figNum = 0;
for mi = 1:numel(metrics)
    metric    = metrics{mi};
    costTitle = metricTitles{mi};
    fprintf('\n=== Cost metric: %s ===\n', costTitle);

    % Per-environment means, accumulated across the environment loop so they can be averaged
    % afterward -- ei rows, one (planner,model) column each, matching plannerNames' order.
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
            tok  = sprintf('m%d_%s_%s', models(plannerModelIdx(pi)), modelDeltaLabel, metric);
            runs = loadRuns(envDir, env, plannerNames{pi}, tok, numRunsPer);
            fprintf('  %-34s %-18s : %d runs\n', plannerNames{pi}, ...
                    ['[' modelNames{plannerModelIdx(pi)} ']'], numel(runs));

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

        % --- % improvement over KPAX, matched model-for-model within this environment. ---
        mPctImprove = NaN(1, nPlanner);
        for mo = 1:numel(models)
            kIdx = find(plannerBaseIdx == kpaxBaseIdx & plannerModelIdx == mo, 1);
            kpaxCost = mFinalCost(kIdx);
            if isnan(kpaxCost) || kpaxCost <= 0, continue; end
            for pi = find(plannerModelIdx == mo)
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
        ylabel(sprintf('%% Improvement over KPAX (Final %s, same model)', costTitle));
        title(sprintf(['Time to First Solution vs %% Improvement over KPAX \x2014 %s, %s\n' ...
                       'marker size = vehicle model (small->large = Double Integrator->Dubins->Quad); ' ...
                       'KPAX itself sits at 0%%'], envTitle, costTitle), 'FontWeight', 'bold');

        modelGroupLabels = cell(1, nPlanner);
        for pi = 1:nPlanner
            modelGroupLabels{pi} = modelNames{plannerModelIdx(pi)};
        end
        writeImprovementTable(plannerDisplay, mFirstSolTime, mFinalCost, mPctImprove, envDir, ...
                              sprintf('paper_benchmark_improvement_%s_%s.csv', sanitize_name(env), metric), ...
                              modelGroupLabels, 'Model');
    end

    % --- LEVEL 2 (intermediate): per-model average, across environments only. Mean of the
    % per-environment MEANS computed above, not a pooled re-mean over every run. Stored as
    % (model x baseName) so LEVEL 3 below can average THESE (not the raw per-environment grid),
    % for the same "mean of means" reason, one level up. ---
    modelAvgFirstSolTime = NaN(numel(models), numel(baseNames));
    modelAvgFinalCost    = NaN(numel(models), numel(baseNames));
    modelAvgPctImprove   = NaN(numel(models), numel(baseNames));

    for mo = 1:numel(models)
        for si = 1:numel(baseNames)
            colIdx = find(plannerBaseIdx == si & plannerModelIdx == mo, 1);
            modelAvgFirstSolTime(mo, si) = mean(allFirstSolTime(:, colIdx), 1, 'omitnan');
            modelAvgFinalCost(mo, si)    = mean(allFinalCost(:, colIdx), 1, 'omitnan');
            modelAvgPctImprove(mo, si)   = mean(allPctImprove(:, colIdx), 1, 'omitnan');
        end

        figNum = figNum + 1;
        figure('Name', sprintf('%s Average - Improvement Scatter (%s)', modelTitles{mo}, costTitle), ...
               'Position', [140 + 20 * mo, 140 + 15 * mo, 1180, 700]);
        improvementScatter(modelAvgFirstSolTime(mo, :), modelAvgPctImprove(mo, :), baseMarkers, ...
                           baseColors, modelMarkerSizes(mo) * ones(1, numel(baseNames)), baseDisplay);
        xlabel('Avg Time to First Solution (ms)');
        ylabel(sprintf('%% Improvement over KPAX (Final %s)', costTitle));
        title(sprintf(['Time to First Solution vs %% Improvement over KPAX \x2014 %s, Averaged ' ...
                       'Across All Environments, %s\nKPAX itself sits at 0%%'], modelTitles{mo}, costTitle), ...
              'FontWeight', 'bold');

        writeImprovementTable(baseDisplay, modelAvgFirstSolTime(mo, :), modelAvgFinalCost(mo, :), ...
                              modelAvgPctImprove(mo, :), baseDir, ...
                              sprintf('paper_benchmark_improvement_%s_average_%s.csv', modelNames{mo}, metric));
    end

    % --- LEVEL 3 (grand average): mean of the LEVEL 2 per-model averages, across models -- again
    % mean of means, not a pooled re-mean over the raw (environment x model) grid, so a model with
    % more successful environments does not implicitly outweigh one with fewer. FinalCost is NOT
    % averaged here (see top-of-file comment) -- it has no single physical unit once blended
    % across vehicle models with different workspace scales. ---
    grandFirstSolTime = mean(modelAvgFirstSolTime, 1, 'omitnan');
    grandPctImprove   = mean(modelAvgPctImprove, 1, 'omitnan');

    figNum = figNum + 1;
    figure('Name', sprintf('Grand Average - Improvement Scatter (%s)', costTitle), ...
           'Position', [180, 180, 1180, 700]);
    improvementScatter(grandFirstSolTime, grandPctImprove, baseMarkers, baseColors, ...
                       GRAND_MARKER_SIZE * ones(1, numel(baseNames)), baseDisplay);
    xlabel('Avg Time to First Solution (ms)');
    ylabel(sprintf('%% Improvement over KPAX (Final %s)', costTitle));
    title(sprintf(['Time to First Solution vs %% Improvement over KPAX \x2014 Grand Average, ' ...
                   'All Models x All Environments, %s\nKPAX itself sits at 0%%'], costTitle), ...
          'FontWeight', 'bold');

    writeImprovementTable(baseDisplay, grandFirstSolTime, NaN(1, numel(baseNames)), grandPctImprove, ...
                          baseDir, sprintf('paper_benchmark_improvement_grand_average_%s.csv', metric));
end

fprintf('\nGenerated %d figures.\n', figNum);

%% ======================================================================
%% Helper functions (local functions)
%% ======================================================================

function improvementScatter(x, y, markers, colors, sizes, labels)
    % One point per series; colour+marker shape = planner identity, marker SIZE = model (or a
    % single uniform size once model has been averaged away -- see call sites). A y=0 reference
    % line is the "no improvement over KPAX" anchor -- KPAX's own points sit on it by construction.
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

function writeImprovementTable(plannerLabels, firstSolTimeVals, finalCostVals, pctImproveVals, ...
                               outDir, csvName, groupLabels, groupColName)
    % groupLabels/groupColName are OPTIONAL -- the per-environment table passes a per-row Model
    % label (12 rows: 4 planners x 3 models); the per-model-average and grand-average tables omit
    % them (4 rows: one per planner, since model is already fixed or fully averaged away by the
    % caller). Same column schema at every level (including FinalCost, all-NaN at the grand level)
    % so the three tiers of CSV stay easy to compare/concatenate.
    if nargin < 7, groupLabels = {}; end
    if nargin < 8, groupColName = 'Group'; end
    hasGroup = ~isempty(groupLabels);

    n = numel(plannerLabels);
    tPlanner = plannerLabels(:);
    if hasGroup
        tGroup = groupLabels(:);
    end

    if hasGroup
        fprintf('%-55s %-18s %16s %14s %16s\n', 'Planner', groupColName, 'TimeToFirst(ms)', 'FinalCost', 'PctImproveKPAX');
    else
        fprintf('%-55s %16s %14s %16s\n', 'Planner', 'TimeToFirst(ms)', 'FinalCost', 'PctImproveKPAX');
    end
    for pi = 1:n
        if hasGroup
            fprintf('%-55s %-18s %16.2f %14.4f %16.2f\n', tPlanner{pi}, tGroup{pi}, ...
                    firstSolTimeVals(pi), finalCostVals(pi), pctImproveVals(pi));
        else
            fprintf('%-55s %16.2f %14.4f %16.2f\n', tPlanner{pi}, ...
                    firstSolTimeVals(pi), finalCostVals(pi), pctImproveVals(pi));
        end
    end

    if hasGroup
        T = table(tPlanner, tGroup, firstSolTimeVals(:), finalCostVals(:), pctImproveVals(:), ...
                  'VariableNames', {'Planner', groupColName, 'TimeToFirst_ms', 'FinalCost', 'PctImprovementOverKPAX'});
    else
        T = table(tPlanner, firstSolTimeVals(:), finalCostVals(:), pctImproveVals(:), ...
                  'VariableNames', {'Planner', 'TimeToFirst_ms', 'FinalCost', 'PctImprovementOverKPAX'});
    end
    writetable(T, fullfile(outDir, csvName));
    fprintf('Table written to: %s\n', csvName);
end

function runs = loadRuns(dataDir, env, planner, delta, numRuns)
    % Load one (planner, delta) series' per-run CSVs; missing files are skipped. Verbatim copy of
    % process_paper_benchmark_v2_and_plot.m's loadRuns() -- keep both in step by hand. `delta` here
    % is the FULL on-disk token (e.g. "m1_tiny_length"), not just the bare delta label.
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
    % process_paper_benchmark_v2_and_plot.m's version.
    s = regexprep(s, '[^a-zA-Z0-9_-]', '_');
end

function t = firstSolTime(tbl, thresh)
    % Elapsed time (ms) at which this run first reached a finite (< thresh) best_cost;
    % -1 if it never found a solution. Verbatim copy of process_paper_benchmark_v2_and_plot.m's.
    solIdx = find(tbl.best_cost < thresh, 1, 'first');
    if isempty(solIdx), t = -1; else, t = tbl.elapsed_time_ms(solIdx); end
end

function c = finalCost(tbl, thresh)
    % Verbatim copy of process_paper_benchmark_v2_and_plot.m's finalCost().
    costs = tbl.best_cost;
    costs(costs > thresh) = NaN;
    v = costs(~isnan(costs));
    if isempty(v), c = NaN; else, c = v(end); end
end
