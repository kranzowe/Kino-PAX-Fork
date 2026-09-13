%% Paper Benchmark Improvement Scatter - grand-average improvement over Kino-PAX+, both axis styles
% Reads per-iteration CSVs produced by examples/gpu/paper_benchmark_v2.cu (run via
% scripts/run_paper_benchmark_v2.sh) -- SEPARATE from process_paper_benchmark_v2_and_plot.m, which
% plots one vehicle model's data at a time. This script reads ALL FOUR environments AND ALL THREE
% vehicle models in one pass (see `environments`/`models` below).
%
% ONLY THE GRAND AVERAGE IS PLOTTED (across ALL environments AND ALL models) -- per-environment and
% per-model numbers are still COMPUTED and WRITTEN TO CSV at every level (for reference), just not
% plotted as their own figures any more; there used to be a figure per environment and per model
% too, which is what made this a 16-figure script. Averaging is MEAN OF MEANS at every step, never
% a pooled re-mean over raw runs -- an environment/model with a lower success rate does not get
% implicitly downweighted just for having fewer successful runs contributing to its own mean: per-
% run -> per-environment mean -> per-model mean (across environments) -> grand mean (across models).
%
% TWO GRAND-AVERAGE FIGURES PER COST METRIC, both sharing the same Y-axis (% improvement in final
% cost over Kino-PAX+) but with a different X-axis:
%   A) Avg Time to First Solution, in raw milliseconds.
%   B) Time-to-First-Solution SPEEDUP over Kino-PAX+, as a multiple (1x/2x/3x/...) -- computed the
%      same way as the cost metric (per (environment, model) cell, relative to Kino-PAX+'s OWN mean
%      time in that same cell, then averaged the same mean-of-means way). Raw milliseconds are not
%      comparable across environments (a fast environment and a slow one shouldn't be pooled
%      directly) any more than raw cost is comparable across vehicle models -- expressing time as a
%      speedup MULTIPLE over Kino-PAX+ normalizes that away, the same fix % improvement already
%      applies to cost, which is why averaging it across all four environments is meaningful in a
%      way raw milliseconds isn't.
% Both figures carry Y error bars (+/-1 std of the 3 per-model % improvement averages that feed the
% grand mean -- "how much does this planner's improvement vary from one vehicle model to the
% next"); figure (B) ALSO carries X error bars, computed the identical way for the speedup metric.
%
% % IMPROVEMENT / SPEEDUP is computed against KINO-PAX+'S OWN mean value AT THAT SAME MODEL, in
% that same environment (and, cost metric) -- not against a single global Kino-PAX+ number --
% because Kino-PAX+'s own performance already varies by vehicle model, so comparing every model
% against one Kino-PAX+ baseline would conflate "this planner is better" with "this model made
% Kino-PAX+ worse/slower." Kino-PAX+ itself sits at (0%, 1x) by construction on both figures.
%
% RAW FINAL COST IS NOT REPORTED AS ITS OWN COLUMN/METRIC ANYWHERE IN THIS FILE -- only the
% Kino-PAX+-relative % improvement is. Raw cost is only comparable WITHIN one vehicle model (Quad's
% workspace is [0,100]^3 while Double Integrator/Dubins Airplane are [0,1]^3), so a raw "final cost"
% column would either be meaningless once blended across models or need a separate, inconsistent
% schema per level. NOTE: a NaN in either metric still means something real and is NOT fixable by
% this script -- it means NO run in that (planner, model, environment) cell ever found a solution,
% so there is nothing to compare at all; that reflects the actual benchmark results, not a bug.
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
% on purpose, see e.g. process_countingstars_and_plot.m's own copies). Planner display names/
% colors/markers below are ALSO kept in step by hand with that file's (Kino-PAX/Kino-PAX+/
% SimpleCombo/KinoPax*, its colors, its x/+/o/p markers).
%
% FIG_WIDTH/FIG_HEIGHT (below, in Configuration) size every figure this file produces -- all four
% share one size, since they're all the same grand-average scatter type.
%
% USAGE: cd into the PARENT of the environment folders, then call BY NAME, not via run() -- and NOT
% via the MATLAB Editor's Run/F5 button, which silently cd's to this file's OWN folder first,
% exactly like run() does, even if you already cd'd to the data folder in the Command Window first:
%   cd build/Data/Benchmarks/PaperBenchmarkV2
%   addpath('<repo>/scripts')
%   process_paper_benchmark_improvement_scatter
% If every series reports 0 runs, check the "Running from:" line this script prints at startup --
% that is MATLAB's actual pwd, and it must be the PARENT folder above, not the scripts folder and
% not one specific environment subfolder.

clear; clc; close all;

fprintf('Running from: %s\n', pwd);

%% --- Configuration ---
baseDir = '';   % '' = current directory (run this from Data/Benchmarks/PaperBenchmarkV2, the
                % PARENT of the environment subfolders -- NOT from inside one of them, unlike
                % process_paper_benchmark_v2_and_plot.m)

% Figure size, in pixels -- every figure this file produces shares this one size (both grand-
% average figures, for both cost metrics, are all the same plot type). Tune to taste.
FIG_WIDTH  = 1180;
FIG_HEIGHT = 700;

% ALL FOUR ENVIRONMENTS -- paper_benchmark_v2.cu runs on countingstars_sweep.cu's harness (unlike
% the old paper_benchmark.cu, which hung at tiny/empty and needed `empty` pulled out as a
% diagnostic), so `empty` is included this pass same as the other three.
environments = {'empty', 'house', 'narrowPassage', 'zigzag'};
envTitles    = {'Empty', 'House', 'Narrow Passage', 'Zigzag Corridor (tightened)'};

% Fail fast and clearly if baseDir/environments don't resolve from here, rather than silently
% reporting "0 runs" for every single series -- exactly the confusing symptom this guards against.
missingEnvDirs = {};
for ei = 1:numel(environments)
    d = fullfile(baseDir, environments{ei});
    if ~isfolder(d)
        missingEnvDirs{end + 1} = d; %#ok<SAGROW>
    end
end
if ~isempty(missingEnvDirs)
    fprintf(2, 'ERROR: environment folder(s) not found relative to pwd (%s):\n', pwd);
    for k = 1:numel(missingEnvDirs)
        fprintf(2, '  %s\n', missingEnvDirs{k});
    end
    error(['cd into the PARENT of the environment subfolders (e.g. build/Data/Benchmarks/', ...
           'PaperBenchmarkV2) and call this script BY NAME, not via run() or the Editor''s Run ', ...
           'button -- both silently cd elsewhere first.']);
end

% Cost metric axis -- must match run_paper_benchmark_v2.sh's COST_LABELS.
metrics       = {'length', 'effort'};
metricTitles  = {'Workspace Path Length', 'Control Effort'};

% Model axis -- must match run_paper_benchmark_v2.sh's MODEL_IDS. Each model runs at exactly ONE
% discretization now (modelDeltaLabel) -- see run_paper_benchmark_v2.sh's MODEL_W_R1S/MODEL_C_R1S/
% MODEL_V_R1S for each model's actual region count.
models          = [1, 2, 3];
modelNames      = {'DoubleIntegrator', 'DubinsAirplane', 'Quad'};   % short, filename-safe tokens
modelDeltaLabel = 'tiny';    % all three models currently share this one delta label

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
% Display names, colors, and markers -- kept in step by hand with
% process_paper_benchmark_v2_and_plot.m's (baseNames above are the internal matching tokens
% loadRuns() needs and are unaffected by any of this).
baseDisplay = { ...
    'Kino-PAX', ...
    'Kino-PAX+', ...
    'SimpleCombo', ...
    'KinoPax*' ...
};
% color = planner identity: Kino-PAX near-black, Kino-PAX+ vivid blue, SimpleCombo vivid red,
% KinoPax* vivid yellow -- each "family" reads as its own hue, saturated enough to pop.
baseColors = [ ...
    0.10 0.10 0.10;    % Kino-PAX
    0.00 0.40 0.95;    % Kino-PAX+ (vivid blue)
    0.85 0.05 0.05;    % SimpleCombo (vivid red)
    0.95 0.70 0.00 ];  % KinoPax* (vivid yellow/gold)
% marker = planner identity too: X, +, circle, star -- matches the display names above (Kino-PAX
% is the "X" mark, Kino-PAX+ the "+" mark, SimpleCombo a plain circle, KinoPax* an actual star).
baseMarkers = {'x', '+', 'o', 'p'};
GRAND_MARKER_SIZE = 12;   % no per-point axis left to encode via size once only the grand-average
                          % (model fully collapsed away) is plotted
refBaseIdx = find(strcmp(baseNames, 'KinoPaxPlus'), 1);
if isempty(refBaseIdx)
    error('KinoPaxPlus not found in baseNames -- it is the improvement reference and must be present.');
end

% --- Build the (planner, model) lookup arrays -- planner-major, same convention as
% process_paper_benchmark_v2_and_plot.m. Only identity/indexing fields are kept here (no
% color/marker/size fields) -- those only ever fed the old per-environment/per-model FIGURES,
% which are gone; the grand figures use baseColors/baseMarkers directly instead. ---
plannerNames    = {};
plannerDisplay  = {};
plannerModelIdx = [];   % index into `models`
plannerBaseIdx  = [];   % index into `baseNames`/`baseDisplay`

for si = 1:numel(baseNames)
    for mo = 1:numel(models)
        plannerNames{end + 1}    = baseNames{si};                                      %#ok<SAGROW>
        plannerDisplay{end + 1}  = sprintf('%s [%s]', baseDisplay{si}, modelNames{mo}); %#ok<SAGROW>
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
    allPctImprove   = NaN(numel(environments), nPlanner);
    allTimeMultiple = NaN(numel(environments), nPlanner);

    for ei = 1:numel(environments)
        env    = environments{ei};
        envDir = fullfile(baseDir, env);
        fprintf('\n--- Environment: %s ---\n', env);

        mFirstSolTime = NaN(1, nPlanner);
        mFinalCost    = NaN(1, nPlanner);   % used only to derive mPctImprove below -- not an output
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

        % --- % improvement over Kino-PAX+ (cost) and time-to-first-solution SPEEDUP over
        % Kino-PAX+ (time), both matched model-for-model within this environment. A NaN in either
        % means no run in that (planner, model) cell ever found a solution -- real, not a bug (see
        % top-of-file comment). ---
        mPctImprove   = NaN(1, nPlanner);
        mTimeMultiple = NaN(1, nPlanner);
        for mo = 1:numel(models)
            refIdx  = find(plannerBaseIdx == refBaseIdx & plannerModelIdx == mo, 1);
            refCost = mFinalCost(refIdx);
            refTime = mFirstSolTime(refIdx);
            for pi = find(plannerModelIdx == mo)
                if ~isnan(refCost) && refCost > 0 && ~isnan(mFinalCost(pi))
                    mPctImprove(pi) = 100 * (refCost - mFinalCost(pi)) / refCost;
                end
                if ~isnan(refTime) && refTime > 0 && ~isnan(mFirstSolTime(pi)) && mFirstSolTime(pi) > 0
                    mTimeMultiple(pi) = refTime / mFirstSolTime(pi);
                end
            end
        end

        allFirstSolTime(ei, :) = mFirstSolTime;
        allPctImprove(ei, :)   = mPctImprove;
        allTimeMultiple(ei, :) = mTimeMultiple;

        % Per-environment level is no longer plotted (see top-of-file comment) -- still written to
        % CSV for reference.
        modelGroupLabels = cell(1, nPlanner);
        for pi = 1:nPlanner
            modelGroupLabels{pi} = modelNames{plannerModelIdx(pi)};
        end
        writeImprovementTable(plannerDisplay, mFirstSolTime, mTimeMultiple, mPctImprove, envDir, ...
                              sprintf('paper_benchmark_improvement_%s_%s.csv', sanitize_name(env), metric), ...
                              modelGroupLabels, 'Model');
    end

    % --- Per-model average, across environments only. Mean of the per-environment MEANS computed
    % above, not a pooled re-mean over every run. Stored as (model x baseName) so the grand average
    % below can average THESE (not the raw per-environment grid), for the same "mean of means"
    % reason, one level up; also what the grand-average error bars are computed FROM (std across
    % the 3 rows here, per column). No longer plotted on its own (see top-of-file comment) -- still
    % written to CSV for reference. ---
    modelAvgFirstSolTime = NaN(numel(models), numel(baseNames));
    modelAvgPctImprove   = NaN(numel(models), numel(baseNames));
    modelAvgTimeMultiple = NaN(numel(models), numel(baseNames));

    for mo = 1:numel(models)
        for si = 1:numel(baseNames)
            colIdx = find(plannerBaseIdx == si & plannerModelIdx == mo, 1);
            modelAvgFirstSolTime(mo, si) = mean(allFirstSolTime(:, colIdx), 1, 'omitnan');
            modelAvgPctImprove(mo, si)   = mean(allPctImprove(:, colIdx), 1, 'omitnan');
            modelAvgTimeMultiple(mo, si) = mean(allTimeMultiple(:, colIdx), 1, 'omitnan');
        end

        writeImprovementTable(baseDisplay, modelAvgFirstSolTime(mo, :), modelAvgTimeMultiple(mo, :), ...
                              modelAvgPctImprove(mo, :), baseDir, ...
                              sprintf('paper_benchmark_improvement_%s_average_%s.csv', modelNames{mo}, metric));
    end

    % --- GRAND AVERAGE: mean of the per-model averages, across models -- again mean of means, not
    % a pooled re-mean over the raw (environment x model) grid, so a model with more successful
    % environments does not implicitly outweigh one with fewer. Error bars are +/-1 std ACROSS THE
    % 3 MODELS (i.e. how much the per-model averages that fed this mean actually disagree), not
    % across the underlying runs -- that is the natural "spread" question at this level. ---
    grandFirstSolTime = mean(modelAvgFirstSolTime, 1, 'omitnan');
    grandPctImprove   = mean(modelAvgPctImprove, 1, 'omitnan');
    grandTimeMultiple = mean(modelAvgTimeMultiple, 1, 'omitnan');

    grandPctImproveErr = std(modelAvgPctImprove, 0, 1, 'omitnan');
    grandPctImproveErr(isnan(grandPctImproveErr)) = 0;   % <2 contributing models -> no visible bar
    grandTimeMultipleErr = std(modelAvgTimeMultiple, 0, 1, 'omitnan');
    grandTimeMultipleErr(isnan(grandTimeMultipleErr)) = 0;

    % --- FIGURE A: Avg Time to First Solution (ms) vs % Improvement over Kino-PAX+. Y error bars
    % only (cost) -- x is raw milliseconds, not requested to carry error bars. ---
    figNum = figNum + 1;
    figure('Name', sprintf('Grand Average - Improvement Scatter (%s)', costTitle), ...
           'Position', [180, 180, FIG_WIDTH, FIG_HEIGHT]);
    improvementScatter(grandFirstSolTime, grandPctImprove, baseMarkers, baseColors, ...
                       GRAND_MARKER_SIZE * ones(1, numel(baseNames)), baseDisplay, grandPctImproveErr);
    xlabel('Avg Time to First Solution (ms)');
    ylabel(sprintf('%% Improvement over Kino-PAX+ (Final %s)', costTitle));
    title(sprintf(['Time to First Solution vs %% Improvement over Kino-PAX+ \x2014 Grand Average, ' ...
                   'All Models x All Environments, %s\nerror bars: +/-1 std across the 3 models; ' ...
                   'Kino-PAX+ itself sits at 0%%'], costTitle), 'FontWeight', 'bold');

    % --- FIGURE B: Time-to-First-Solution SPEEDUP over Kino-PAX+ (x, unitless) vs % Improvement
    % over Kino-PAX+. BOTH axes carry error bars now. ---
    figNum = figNum + 1;
    figure('Name', sprintf('Grand Average - Speedup Scatter (%s)', costTitle), ...
           'Position', [220, 220, FIG_WIDTH, FIG_HEIGHT]);
    improvementScatter(grandTimeMultiple, grandPctImprove, baseMarkers, baseColors, ...
                       GRAND_MARKER_SIZE * ones(1, numel(baseNames)), baseDisplay, ...
                       grandPctImproveErr, grandTimeMultipleErr);
    xline(1, 'k--', '1x = same speed as Kino-PAX+', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    xlabel('Time-to-First-Solution Speedup over Kino-PAX+ (x)');
    ylabel(sprintf('%% Improvement over Kino-PAX+ (Final %s)', costTitle));
    title(sprintf(['Speedup vs %% Improvement over Kino-PAX+ \x2014 Grand Average, All Models x ' ...
                   'All Environments, %s\nerror bars: +/-1 std across the 3 models; Kino-PAX+ ' ...
                   'itself sits at (1x, 0%%)'], costTitle), 'FontWeight', 'bold');

    writeImprovementTable(baseDisplay, grandFirstSolTime, grandTimeMultiple, grandPctImprove, ...
                          baseDir, sprintf('paper_benchmark_improvement_grand_average_%s.csv', metric));
end

fprintf('\nGenerated %d figures.\n', figNum);

%% ======================================================================
%% Helper functions (local functions)
%% ======================================================================

function improvementScatter(x, y, markers, colors, sizes, labels, yErr, xErr)
    % One point per series; colour+marker shape = planner identity. yErr (required) and xErr
    % (optional, [] or omitted to skip) are +/- error-bar half-widths per point, drawn UNDER the
    % marker via errorbar() and excluded from the legend (HandleVisibility off) -- see call sites
    % for what each represents (grand-average plots: std across the 3 per-model averages). A y=0
    % reference line is the "no improvement over Kino-PAX+" anchor -- Kino-PAX+'s own points sit on
    % it by construction.
    if nargin < 8, xErr = []; end
    hasXErr = ~isempty(xErr);
    hold on;
    for pi = 1:numel(x)
        if isnan(x(pi)) || isnan(y(pi)), continue; end
        ye = yErr(pi);
        if isnan(ye), ye = 0; end
        if hasXErr
            xe = xErr(pi);
            if isnan(xe), xe = 0; end
            errorbar(x(pi), y(pi), ye, ye, xe, xe, 'LineStyle', 'none', ...
                     'Color', colors(pi, :), 'CapSize', 6, 'HandleVisibility', 'off');
        else
            errorbar(x(pi), y(pi), ye, ye, 'LineStyle', 'none', ...
                     'Color', colors(pi, :), 'CapSize', 6, 'HandleVisibility', 'off');
        end
        m = markers{pi};
        if m == 'x' || m == '+'
            % Stroke-only markers ('x','+') have no fillable interior -- MATLAB ignores
            % MarkerFaceColor for them entirely, so their visible color comes ONLY from
            % MarkerEdgeColor. A much thicker stroke is also what makes a '+' actually read as a
            % bold cross instead of a thin plus sign.
            plot(x(pi), y(pi), m, 'MarkerEdgeColor', colors(pi, :), 'LineWidth', 3.0, ...
                 'MarkerSize', sizes(pi), 'DisplayName', labels{pi});
        else
            plot(x(pi), y(pi), m, ...
                 'MarkerFaceColor', colors(pi, :), ...
                 'MarkerEdgeColor', 'k', 'LineWidth', 0.5, ...
                 'MarkerSize', sizes(pi), 'DisplayName', labels{pi});
        end
    end
    yline(0, 'k--', 'no improvement over Kino-PAX+', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    grid on;
    clickableLegend();
end

function writeImprovementTable(plannerLabels, firstSolTimeVals, timeMultipleVals, pctImproveVals, ...
                               outDir, csvName, groupLabels, groupColName)
    % groupLabels/groupColName are OPTIONAL -- the per-environment table passes a per-row Model
    % label (12 rows: 4 planners x 3 models); the per-model-average and grand-average tables omit
    % them (4 rows: one per planner, since model is already fixed or fully averaged away by the
    % caller). Raw FinalCost is NOT a column here (see top-of-file comment) -- % improvement is
    % already its normalized, cross-model-comparable form, so it is the only cost-related metric
    % kept, at every level; a NaN here means no run in that cell ever found a solution, not a bug.
    if nargin < 7, groupLabels = {}; end
    if nargin < 8, groupColName = 'Group'; end
    hasGroup = ~isempty(groupLabels);

    n = numel(plannerLabels);
    tPlanner = plannerLabels(:);
    if hasGroup
        tGroup = groupLabels(:);
    end

    if hasGroup
        fprintf('%-55s %-18s %16s %14s %20s\n', 'Planner', groupColName, 'TimeToFirst(ms)', ...
                'TimeMultipleX', 'PctImproveKinoPaxPlus');
    else
        fprintf('%-55s %16s %14s %20s\n', 'Planner', 'TimeToFirst(ms)', 'TimeMultipleX', ...
                'PctImproveKinoPaxPlus');
    end
    for pi = 1:n
        if hasGroup
            fprintf('%-55s %-18s %16.2f %14.3f %20.2f\n', tPlanner{pi}, tGroup{pi}, ...
                    firstSolTimeVals(pi), timeMultipleVals(pi), pctImproveVals(pi));
        else
            fprintf('%-55s %16.2f %14.3f %20.2f\n', tPlanner{pi}, ...
                    firstSolTimeVals(pi), timeMultipleVals(pi), pctImproveVals(pi));
        end
    end

    if hasGroup
        T = table(tPlanner, tGroup, firstSolTimeVals(:), timeMultipleVals(:), pctImproveVals(:), ...
                  'VariableNames', {'Planner', groupColName, 'TimeToFirst_ms', ...
                  'TimeMultipleOverKinoPaxPlus', 'PctImprovementOverKinoPaxPlus'});
    else
        T = table(tPlanner, firstSolTimeVals(:), timeMultipleVals(:), pctImproveVals(:), ...
                  'VariableNames', {'Planner', 'TimeToFirst_ms', 'TimeMultipleOverKinoPaxPlus', ...
                  'PctImprovementOverKinoPaxPlus'});
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
