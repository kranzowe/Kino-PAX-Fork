%% Frontier-Growth Comparison + KPAX Frontier-Death Diagnostics
% Reads per-iteration CSVs produced by kinopaxplus_delta_benchmark.cu and
% compares frontier-size growth for KPAX, PruneKPAX, and KinoPaxPlus at matched
% R1 discretizations (grouped by num_regions).
%
% Per-run CSVs (in dataDir):
%   KinoPaxPlus:  {env}_delta{label}_run{n}.csv
%   KPAX:         {env}_KPAX_delta{label}_run{n}.csv
%   PruneKPAX:    {env}_PruneKPAX_delta{label}_run{n}.csv
%
% Per-iteration columns:
%   iteration, frontier_size, tree_size, elapsed_time_ms, best_cost,
%   num_regions, r2_coverage_pct, mean_vertex_score, reactivated
%   (r2_coverage_pct / mean_vertex_score are NaN and reactivated is -1 for KinoPaxPlus)
%
% Figures, per environment:
%   A) frontier_size vs elapsed_time_ms  (one tile per delta, 4 planners, mean +/- std)
%   B) frontier_size vs iteration        (same layout)
%   B2) tree_size vs iteration           (same layout)
%   C) frontier-death diagnostics vs iteration (KPAX + PruneKPAX):
%      r2_coverage_pct, mean_vertex_score, reactivated

clear; clc; close all;

%% --- Configuration ---
dataDir = 'KinoPaxPlusDelta500kmore';   % <-- point at your results folder

environments = {'house'};
envTitles    = {'House'};

% Delta labels must match those used by run_delta_benchmark.sh
deltas   = {'larger', 'large', 'med_large', 'med_small'};

numRuns         = 50;   % KinoPaxPlus runs per delta
numBaselineRuns = 50;   % KPAX / PruneKPAX runs per delta

% Planner styling
colKPAX    = [0.10 0.10 0.10];   % near-black
colPrune   = [0.85 0.45 0.10];   % orange
colKPP     = [0.20 0.40 0.80];   % blue (KinoPaxPlus)
colStar    = [0.55 0.15 0.60];   % purple (KinoPaxSTAR)

MAX_FLOAT_THRESH = 1e30;
numTimeSamples   = 500;

%% ======================================================================
for ei = 1:numel(environments)
    env      = environments{ei};
    envTitle = envTitles{ei};
    fprintf('\n=== Environment: %s ===\n', env);

    % --- Load all runs, grouped by delta and planner ---
    K = struct('kpp', {{}}, 'kpax', {{}}, 'prune', {{}}, 'star', {{}}, 'regions', {[]});
    data = repmat(K, 1, numel(deltas));
    for di = 1:numel(deltas)
        data(di).kpp      = loadRuns(dataDir, env, 'KinoPaxPlus', deltas{di}, numRuns);
        data(di).kpax     = loadRuns(dataDir, env, 'KPAX',        deltas{di}, numBaselineRuns);
        data(di).prune    = loadRuns(dataDir, env, 'PruneKPAX',   deltas{di}, numBaselineRuns);
        data(di).star     = loadRuns(dataDir, env, 'KinoPaxSTAR',    deltas{di}, numRuns);
        data(di).regions  = regionCount(data(di).kpp, data(di).kpax, data(di).prune, data(di).star);
    end

    nD = numel(deltas);
    nCols = 2; nRows = ceil(nD / nCols);

    %% ---------- FIGURE A: frontier vs time ----------
    figure('Name', sprintf('%s - Frontier vs Time', envTitle), 'Position', [40 40 1100 700]);
    tl = tiledlayout(nRows, nCols, 'TileSpacing', 'compact', 'Padding', 'compact');
    for di = 1:nD
        nexttile; hold on;
        tmax = globalMaxTime({data(di).kpax, data(di).prune, data(di).kpp, data(di).star});
        if tmax > 0
            ct = linspace(0, tmax, numTimeSamples);
            plotBandTime(data(di).kpax,     'frontier_size', ct, colKPAX,    '-', 'KPAX');
            plotBandTime(data(di).prune,    'frontier_size', ct, colPrune,   '-', 'PruneKPAX');
            plotBandTime(data(di).kpp,      'frontier_size', ct, colKPP,     '-', 'KinoPaxPlus');
            plotBandTime(data(di).star, 'frontier_size', ct, colStar, '-', 'KinoPaxSTAR');
        end
        title(sprintf('%s  (R1=%s)', deltas{di}, regStr(data(di).regions)), 'Interpreter', 'none');
        xlabel('Elapsed Time (ms)'); ylabel('Frontier Size'); grid on;
        if di == 1, legend('Location', 'best', 'FontSize', 7); end
    end
    title(tl, sprintf('Frontier Size vs Time \x2014 %s', envTitle), 'FontWeight', 'bold');

    %% ---------- FIGURE B: frontier vs iteration ----------
    figure('Name', sprintf('%s - Frontier vs Iteration', envTitle), 'Position', [80 60 1100 700]);
    tl = tiledlayout(nRows, nCols, 'TileSpacing', 'compact', 'Padding', 'compact');
    for di = 1:nD
        nexttile; hold on;
        plotBandIter(data(di).kpax,     'frontier_size', colKPAX,    '-', 'KPAX');
        plotBandIter(data(di).prune,    'frontier_size', colPrune,   '-', 'PruneKPAX');
        plotBandIter(data(di).kpp,      'frontier_size', colKPP,     '-', 'KinoPaxPlus');
        plotBandIter(data(di).star, 'frontier_size', colStar, '-', 'KinoPaxSTAR');
        title(sprintf('%s  (R1=%s)', deltas{di}, regStr(data(di).regions)), 'Interpreter', 'none');
        xlabel('Iteration'); ylabel('Frontier Size'); grid on;
        if di == 1, legend('Location', 'best', 'FontSize', 7); end
    end
    title(tl, sprintf('Frontier Size vs Iteration \x2014 %s', envTitle), 'FontWeight', 'bold');

    %% ---------- FIGURE B2: tree size vs iteration ----------
    figure('Name', sprintf('%s - Tree Size vs Iteration', envTitle), 'Position', [100 80 1100 700]);
    tl = tiledlayout(nRows, nCols, 'TileSpacing', 'compact', 'Padding', 'compact');
    for di = 1:nD
        nexttile; hold on;
        plotBandIter(data(di).kpax,     'tree_size', colKPAX,    '-', 'KPAX');
        plotBandIter(data(di).prune,    'tree_size', colPrune,   '-', 'PruneKPAX');
        plotBandIter(data(di).kpp,      'tree_size', colKPP,     '-', 'KinoPaxPlus');
        plotBandIter(data(di).star, 'tree_size', colStar, '-', 'KinoPaxSTAR');
        title(sprintf('%s  (R1=%s)', deltas{di}, regStr(data(di).regions)), 'Interpreter', 'none');
        xlabel('Iteration'); ylabel('Tree Size'); grid on;
        if di == 1, legend('Location', 'best', 'FontSize', 7); end
    end
    title(tl, sprintf('Tree Size vs Iteration \x2014 %s', envTitle), 'FontWeight', 'bold');

    %% ---------- FIGURE C: frontier-death diagnostics (KPAX + PruneKPAX) ----------
    % Rows = delta, Cols = [R2 coverage %, mean vertex score, reactivated]
    figure('Name', sprintf('%s - Frontier-Death Diagnostics', envTitle), 'Position', [120 40 1200 760]);
    tl = tiledlayout(nD, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    diagCols   = {'r2_coverage_pct', 'mean_vertex_score', 'reactivated'};
    diagTitles = {'R2 coverage (%)', 'mean vertex score', 'reactivated / iter'};
    for di = 1:nD
        for ci = 1:3
            nexttile; hold on;
            plotBandIter(data(di).kpax,     diagCols{ci}, colKPAX,    '-',  'KPAX');
            plotBandIter(data(di).prune,    diagCols{ci}, colPrune,   '--', 'PruneKPAX');
            plotBandIter(data(di).star, diagCols{ci}, colStar, ':',  'KinoPaxSTAR');
            if ci == 1, ylabel(sprintf('%s\nR1=%s', deltas{di}, regStr(data(di).regions)), 'Interpreter', 'none'); end
            if di == 1, title(diagTitles{ci}); end
            if di == nD, xlabel('Iteration'); end
            grid on;
            if di == 1 && ci == 3, legend('Location', 'best', 'FontSize', 7); end
        end
    end
    title(tl, sprintf('Frontier-Death Diagnostics (KPAX / PruneKPAX / KinoPaxSTAR) \x2014 %s', envTitle), 'FontWeight', 'bold');
end

fprintf('\nDone.\n');

%% ====================== helper functions ======================

function runs = loadRuns(dataDir, env, planner, delta, numRuns)
    runs = {};
    for ri = 0:(numRuns - 1)
        switch planner
            case 'KinoPaxPlus'
                fn = sprintf('%s_delta%s_run%d.csv', env, delta, ri);
            case 'KPAX'
                fn = sprintf('%s_KPAX_delta%s_run%d.csv', env, delta, ri);
            case 'PruneKPAX'
                fn = sprintf('%s_PruneKPAX_delta%s_run%d.csv', env, delta, ri);
            case 'KinoPaxSTAR'
                fn = sprintf('%s_KinoPaxSTAR_delta%s_run%d.csv', env, delta, ri);
            otherwise
                error('unknown planner %s', planner);
        end
        fp = fullfile(dataDir, fn);
        if isfile(fp)
            runs{end + 1} = readtable(fp); %#ok<AGROW>
        end
    end
end

function r = regionCount(varargin)
    % first non-empty num_regions found across the supplied run-cell-arrays
    r = NaN;
    for k = 1:numel(varargin)
        runs = varargin{k};
        for ri = 1:numel(runs)
            if ~isempty(runs{ri}) && any(strcmp('num_regions', runs{ri}.Properties.VariableNames))
                r = runs{ri}.num_regions(1);
                return;
            end
        end
    end
end

function s = regStr(r)
    if isnan(r), s = '?'; else, s = sprintf('%d', r); end
end

function tmax = globalMaxTime(groups)
    tmax = 0;
    for g = 1:numel(groups)
        runs = groups{g};
        for ri = 1:numel(runs)
            if ~isempty(runs{ri})
                tmax = max(tmax, max(runs{ri}.elapsed_time_ms));
            end
        end
    end
end

function plotBandIter(runs, col, color, style, name)
    % mean +/- std of column 'col' vs iteration across runs
    if isempty(runs), return; end
    maxIter = 0;
    for ri = 1:numel(runs)
        if ~isempty(runs{ri}), maxIter = max(maxIter, max(runs{ri}.iteration)); end
    end
    if maxIter == 0, return; end
    A = NaN(numel(runs), maxIter);
    for ri = 1:numel(runs)
        if isempty(runs{ri}), continue; end
        it = runs{ri}.iteration;
        v  = getCol(runs{ri}, col);
        if isempty(v), continue; end
        A(ri, it) = v;
    end
    A = sanitize(A);
    drawBand(1:maxIter, mean(A, 1, 'omitnan'), std(A, 0, 1, 'omitnan'), color, style, name);
end

function plotBandTime(runs, col, commonTime, color, style, name)
    % mean +/- std of column 'col' vs a shared time grid (previous-sample hold)
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
            A(ri, :) = interp1(t, v, commonTime, 'previous', NaN);
        end
    end
    A = sanitize(A);
    drawBand(commonTime, mean(A, 1, 'omitnan'), std(A, 0, 1, 'omitnan'), color, style, name);
end

function v = getCol(tbl, col)
    if any(strcmp(col, tbl.Properties.VariableNames))
        v = tbl.(col);
    else
        v = [];
    end
end

function A = sanitize(A)
    % drop KinoPaxPlus sentinels so they don't distort diagnostic means
    A(A > 1e30)  = NaN;   % MAX_FLOAT best_cost, if ever passed here
    A(A == -1)   = NaN;   % reactivated sentinel for planners without it
end

function drawBand(x, mu, sd, color, style, name)
    valid = ~isnan(mu);
    if ~any(valid), return; end
    xv = x(valid); mv = mu(valid); sv = sd(valid); sv(isnan(sv)) = 0;
    fill([xv, fliplr(xv)], [mv + sv, fliplr(mv - sv)], color, ...
        'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(xv, mv, style, 'Color', color, 'LineWidth', 1.8, 'DisplayName', name);
end
