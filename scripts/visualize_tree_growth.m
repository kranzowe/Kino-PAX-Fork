%% Tree-Growth & R1-Region Density Visualization (3D) — all variants
% Renders, per planner variant, the 3D growth of the search tree and a 3D
% heatmap of node density per R1 workspace region, plus a cross-variant
% comparison. Explains *why* variants differ: broad exploration (KPAX) vs
% concentrated cost-optimization (KinoPaxPlus) vs pruning-lean STAR variants.
%
% DATA SOURCE (produced by kinopaxstar_large_benchmark.cu run with --dump-viz,
% i.e. `bash scripts/run_large_benchmark.sh`):
%   viz/meta.csv                     grid + bounds + start/goal (one numeric row)
%   viz/{env}_{variant}_tree.csv     run-0 full tree: idx,x,y,z,vx,vy,vz,parent,cost
% Per-iteration CSVs (in dataDir, optional) give tree_size per iteration so the
% growth checkpoints map to real iterations; falls back to node-count fractions.
%
% Node idx == insertion order for every variant (the tree only appends; pruning
% tombstones nodes in place), so growth is reconstructed by slicing nodes
% [0 .. N) with N increasing. Density per R1 workspace cell reproduces
% getRegion(): floor(W_R1_LENGTH * (coord - W_MIN) / (W_MAX - W_MIN)) per axis,
% marginalized over velocity.

clear; clc; close all;

%% ===================== Configuration =====================
% Paths are resolved relative to this script by default (repo/scripts/..).
dataDir      = '';
vizDir       = 'viz\';
obstacleFile = 'obstacles.csv';

env   = 'house';   % environment name used in the dumped filenames
delta = 'large';   % build delta label used in the per-iteration filenames

env   = 'house';   % environment name used in the dumped filenames
delta = 'large';   % build delta label used in the per-iteration filenames

% Variants to visualize (CSV token = dump filename token). Trim as desired.
variants = { ...
    'KPAX', ...
    'PruneKPAX', ...
    'KinoPaxPlus', ...
    'KinoPaxSTARNoPrune', ...
    'KinoPaxSTARNoPruneNoSpatialHash', ...
    'KinoPaxSTAR', ...
    'KinoPaxSTARcostprune_cap0', ...
    'KinoPaxSTARcostprune_cap20', ...
    'KinoPaxSTARcostprune_cap40', ...
    'KinoPaxSTARcostprune_cap80', ...
    'KinoPaxSTARcostprune_cap100'};

% Pretty labels for titles (same order as `variants`).
variantLabels = { ...
    'KPAX', 'PruneKPAX', 'KinoPaxPlus', ...
    'STARNoPrune', 'STARNoPrune-NoSH', 'KinoPaxSTAR', ...
    'CostPrune cap0', 'CostPrune cap0.2', 'CostPrune cap0.4', ...
    'CostPrune cap0.8', 'CostPrune cap1.0'};

% Per-variant + growth figures render only this core subset (all 11 still appear
% in the cross-variant montage). Swap the cost-prune cap here as desired.
coreVariants = { ...
    'KPAX', 'PruneKPAX', 'KinoPaxPlus', ...
    'KinoPaxSTARNoPrune', 'KinoPaxSTAR', 'KinoPaxSTARcostprune_cap20'};

% --- Toggles ---
PER_VARIANT_FIGS = true;    % one [tree | density] figure per variant
GROWTH_PANELS    = true;    % per-variant density small-multiples over growth
CROSS_COMPARE    = true;    % single montage of final density, shared color scale
DENSITY_STYLE    = 'scatter';  % 'scatter' (marker per R1 cell) or 'voxel' (cubes)
DRAW_OBSTACLES   = true;    % overlay obstacle boxes / wireframe
EDGE_SUBSAMPLE   = 20000;   % max parent edges drawn in the tree plot (0 = none)
NODE_SUBSAMPLE   = 60000;   % max nodes drawn in the tree scatter (perf guard)
GROWTH_NCHK      = 6;       % number of growth checkpoints
SAVE_FIGS        = false;   % also save PNGs to <dataDir>/viz/figs
CMAP             = pickColormap();   % turbo if available, else parula

%% ===================== Load shared data =====================
assert(isfile(fullfile(vizDir, 'meta.csv')), ...
    'meta.csv not found in %s. Run the benchmark with --dump-viz first.', vizDir);
M = readMeta(fullfile(vizDir, 'meta.csv'));
fprintf('Grid: %dx%dx%d workspace R1 cells over [%.2f, %.2f]^3\n', ...
    M.W_R1_LENGTH, M.W_R1_LENGTH, M.W_R1_LENGTH, M.W_MIN, M.W_MAX);

obstacles = [];
if DRAW_OBSTACLES && isfile(obstacleFile)
    obstacles = readmatrix(obstacleFile);
end

figsDir = fullfile(vizDir, 'figs');
if SAVE_FIGS && ~exist(figsDir, 'dir'); mkdir(figsDir); end

% First pass: load every variant's tree + density; track a global max count so
% the cross-variant comparison uses one shared color scale.
nV       = numel(variants);
trees    = cell(1, nV);
vols     = cell(1, nV);
globalMax = 1;
for vi = 1:nV
    treePath = fullfile(vizDir, sprintf('%s_%s_tree.csv', env, variants{vi}));
    if ~isfile(treePath)
        fprintf('  [skip] %-32s (no tree CSV)\n', variants{vi});
        continue;
    end
    T = readtable(treePath);
    trees{vi} = T;
    vols{vi}  = binDensity(T.x, T.y, T.z, M);
    globalMax = max(globalMax, max(vols{vi}(:)));
    fprintf('  [ok]   %-32s %7d nodes, peak cell = %d\n', ...
        variants{vi}, height(T), max(vols{vi}(:)));
end
clim = [1, max(globalMax, 2)];   % shared color limits (log scale) for density

% Indices (into `variants`) of the core subset that gets per-variant + growth figures.
coreIdx = [];
for c = 1:numel(coreVariants)
    j = find(strcmp(variants, coreVariants{c}), 1);
    if ~isempty(j); coreIdx(end + 1) = j; end %#ok<AGROW>
end

%% ===================== Per-variant figures (core subset) =====================
if PER_VARIANT_FIGS
    for vi = coreIdx
        if isempty(trees{vi}); continue; end
        T = trees{vi}; vol = vols{vi};

        figure('Name', sprintf('%s — tree & density', variantLabels{vi}), ...
               'Position', [60 60 1400 640], 'Color', 'w');

        % --- (a) 3D tree growth (colored by insertion order) ---
        ax1 = subplot(1, 2, 1); hold(ax1, 'on');
        drawWorkspaceCube(ax1, M);
        if DRAW_OBSTACLES; drawObstacles(ax1, obstacles, 0.10); end
        drawTree(ax1, T, EDGE_SUBSAMPLE, NODE_SUBSAMPLE, CMAP);
        drawStartGoal(ax1, M);
        finishAxes(ax1, sprintf('%s — tree growth (color = insertion order)', variantLabels{vi}), M);
        cb1 = colorbar(ax1); cb1.Label.String = 'node index (early \rightarrow late)';

        % --- (b) 3D density heatmap ---
        ax2 = subplot(1, 2, 2); hold(ax2, 'on');
        drawWorkspaceCube(ax2, M);
        if DRAW_OBSTACLES; drawObstacles(ax2, obstacles, 0.06); end
        drawDensity(ax2, vol, M, clim, DENSITY_STYLE, CMAP);
        drawStartGoal(ax2, M);
        finishAxes(ax2, sprintf('%s — R1 region density', variantLabels{vi}), M);
        cb2 = colorbar(ax2); cb2.Label.String = 'nodes per R1 cell';

        if SAVE_FIGS
            saveas(gcf, fullfile(figsDir, sprintf('%s_tree_density.png', variants{vi})));
        end
    end
end

%% ===================== Growth small-multiples (core subset) =====================
if GROWTH_PANELS
    for vi = coreIdx
        if isempty(trees{vi}); continue; end
        T = trees{vi};
        [chkN, chkLbl] = growthCheckpoints(dataDir, env, variants{vi}, delta, ...
                                           height(T), GROWTH_NCHK);

        figure('Name', sprintf('%s — density growth', variantLabels{vi}), ...
               'Position', [80 80 1500 560], 'Color', 'w');
        tl = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');
        for c = 1:numel(chkN)
            N   = chkN(c);
            vol = binDensity(T.x(1:N), T.y(1:N), T.z(1:N), M);
            ax  = nexttile(tl); hold(ax, 'on');
            drawWorkspaceCube(ax, M);
            drawDensity(ax, vol, M, clim, DENSITY_STYLE, CMAP);
            finishAxes(ax, chkLbl{c}, M);
            view(ax, 2);   % top-down reads cleanest for a strip of panels
        end
        cb = colorbar(ax); cb.Label.String = 'nodes / R1 cell';
        title(tl, sprintf('%s — tree density growth', variantLabels{vi}), ...
              'FontWeight', 'bold');
        if SAVE_FIGS
            saveas(gcf, fullfile(figsDir, sprintf('%s_growth.png', variants{vi})));
        end
    end
end

%% ===================== Cross-variant comparison =====================
if CROSS_COMPARE
    present = find(~cellfun(@isempty, vols));
    if ~isempty(present)
        figure('Name', 'Cross-variant density comparison', ...
               'Position', [40 40 1600 900], 'Color', 'w');
        tl = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');
        for vi = present
            ax = nexttile(tl); hold(ax, 'on');
            drawWorkspaceCube(ax, M);
            if DRAW_OBSTACLES; drawObstacles(ax, obstacles, 0.05); end
            drawDensity(ax, vols{vi}, M, clim, DENSITY_STYLE, CMAP);
            drawStartGoal(ax, M);
            finishAxes(ax, variantLabels{vi}, M);
        end
        cb = colorbar(ax); cb.Label.String = 'nodes per R1 cell';
        title(tl, sprintf('R1-region node density — %s (shared log color scale)', env), ...
              'FontWeight', 'bold');
        if SAVE_FIGS
            saveas(gcf, fullfile(figsDir, 'cross_variant_density.png'));
        end
    end
end

fprintf('\nDone.\n');

%% ============================ helpers ============================

function M = readMeta(path)
    T = readtable(path);
    M = table2struct(T);   % single-row table -> struct with named numeric fields
end

function vol = binDensity(x, y, z, M)
    % Bin workspace coords into the W_R1_LENGTH^3 grid, reproducing getRegion().
    L = M.W_R1_LENGTH;
    ix = clampIdx(floor(L * (x - M.W_MIN) / (M.W_MAX - M.W_MIN)), L);
    iy = clampIdx(floor(L * (y - M.W_MIN) / (M.W_MAX - M.W_MIN)), L);
    iz = clampIdx(floor(L * (z - M.W_MIN) / (M.W_MAX - M.W_MIN)), L);
    vol = accumarray([ix iy iz] + 1, 1, [L L L]);
end

function idx = clampIdx(idx, L)
    idx(idx < 0)      = 0;
    idx(idx > L - 1)  = L - 1;
end

function drawDensity(ax, vol, M, clim, style, cmap)
    % 3D density of occupied R1 cells: 'scatter' (marker per cell) or 'voxel' (cube).
    L  = M.W_R1_LENGTH;
    cs = (M.W_MAX - M.W_MIN) / L;                 % cell size
    [ii, jj, kk] = ind2sub(size(vol), find(vol > 0));
    v  = vol(sub2ind(size(vol), ii, jj, kk));
    cx = M.W_MIN + (ii - 0.5) * cs;               % cell centers (1-indexed -> center)
    cy = M.W_MIN + (jj - 0.5) * cs;
    cz = M.W_MIN + (kk - 0.5) * cs;

    colormap(ax, cmap);
    caxis(ax, clim);
    try, set(ax, 'ColorScale', 'log'); catch, end   % log color (R2019b+)

    if strcmpi(style, 'voxel')
        cmax = max(clim(2), 2);
        for n = 1:numel(v)
            a = 0.05 + 0.85 * (log(v(n) + 1) / log(cmax + 1));   % alpha ~ log density
            col = val2rgb(v(n), clim, cmap);
            drawCube(ax, [cx(n) cy(n) cz(n)], cs, col, a);
        end
    else  % scatter (default): one marker per occupied R1 cell
        cmax = max(clim(2), 2);
        sz = 10 + 190 * (log(v + 1) / log(cmax + 1));            % size ~ log density
        scatter3(ax, cx, cy, cz, sz, v, 'filled', ...
                 'MarkerFaceAlpha', 0.75, 'MarkerEdgeColor', 'none');
    end
end

function drawTree(ax, T, maxEdges, maxNodes, cmap)
    % Parent edges (NaN-separated segments) + nodes colored by insertion order.
    p = T.parent;                       % 0-indexed parent, -1 for root
    child = find(p >= 0);
    if maxEdges > 0 && numel(child) > maxEdges
        child = child(round(linspace(1, numel(child), maxEdges)));
    end
    if ~isempty(child)
        pr = p(child) + 1;              % parent row (idx is 0-based, rows in idx order)
        X = [T.x(child)'; T.x(pr)'; nan(1, numel(child))];
        Y = [T.y(child)'; T.y(pr)'; nan(1, numel(child))];
        Z = [T.z(child)'; T.z(pr)'; nan(1, numel(child))];
        plot3(ax, X(:), Y(:), Z(:), '-', 'Color', [0.72 0.72 0.75], 'LineWidth', 0.3);
    end

    n = height(T);
    ridx = 1:n;
    if maxNodes > 0 && n > maxNodes
        ridx = round(linspace(1, n, maxNodes));
    end
    colormap(ax, cmap);
    scatter3(ax, T.x(ridx), T.y(ridx), T.z(ridx), 6, T.idx(ridx), 'filled', ...
             'MarkerFaceAlpha', 0.7);
end

function drawStartGoal(ax, M)
    plot3(ax, M.start_x, M.start_y, M.start_z, 'o', 'MarkerSize', 11, ...
          'MarkerFaceColor', [0.1 0.7 0.1], 'MarkerEdgeColor', 'k', 'LineWidth', 1);
    plot3(ax, M.goal_x, M.goal_y, M.goal_z, 'p', 'MarkerSize', 16, ...
          'MarkerFaceColor', [0.9 0.2 0.2], 'MarkerEdgeColor', 'k', 'LineWidth', 1);
end

function drawWorkspaceCube(ax, M)
    a = M.W_MIN; b = M.W_MAX;
    V = [a a a; b a a; b b a; a b a; a a b; b a b; b b b; a b b];
    E = [1 2;2 3;3 4;4 1; 5 6;6 7;7 8;8 5; 1 5;2 6;3 7;4 8];
    for k = 1:size(E, 1)
        plot3(ax, V(E(k, :), 1), V(E(k, :), 2), V(E(k, :), 3), 'k-', 'LineWidth', 0.4);
    end
end

function drawObstacles(ax, obstacles, alpha)
    if isempty(obstacles); return; end
    faces = [1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8; 1 2 3 4; 5 6 7 8];
    for j = 1:size(obstacles, 1)
        o = obstacles(j, :);   % xmin ymin zmin xmax ymax zmax
        V = [o(1) o(2) o(3); o(4) o(2) o(3); o(4) o(5) o(3); o(1) o(5) o(3); ...
             o(1) o(2) o(6); o(4) o(2) o(6); o(4) o(5) o(6); o(1) o(5) o(6)];
        patch(ax, 'Vertices', V, 'Faces', faces, 'FaceColor', [0.5 0.5 0.55], ...
              'EdgeColor', [0.3 0.3 0.3], 'FaceAlpha', alpha, 'EdgeAlpha', alpha, ...
              'HandleVisibility', 'off');
    end
end

function drawCube(ax, center, cs, col, alpha)
    a = cs / 2;
    o = center - a;  h = center + a;
    V = [o(1) o(2) o(3); h(1) o(2) o(3); h(1) h(2) o(3); o(1) h(2) o(3); ...
         o(1) o(2) h(3); h(1) o(2) h(3); h(1) h(2) h(3); o(1) h(2) h(3)];
    F = [1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8; 1 2 3 4; 5 6 7 8];
    patch(ax, 'Vertices', V, 'Faces', F, 'FaceColor', col, 'EdgeColor', 'none', ...
          'FaceAlpha', alpha, 'HandleVisibility', 'off');
end

function finishAxes(ax, ttl, M)
    axis(ax, 'equal');
    xlim(ax, [M.W_MIN M.W_MAX]); ylim(ax, [M.W_MIN M.W_MAX]); zlim(ax, [M.W_MIN M.W_MAX]);
    xlabel(ax, 'x'); ylabel(ax, 'y'); zlabel(ax, 'z');
    view(ax, 3); grid(ax, 'on'); box(ax, 'on');
    title(ax, ttl, 'Interpreter', 'none', 'FontWeight', 'bold', 'FontSize', 9);
end

function [chkN, lbl] = growthCheckpoints(dataDir, env, token, delta, nNodes, K)
    % Node-count checkpoints for the growth panels. If the run-0 per-iteration
    % CSV is found, checkpoints land on real iterations (labels show iter + N);
    % otherwise fall back to evenly spaced node-count fractions.
    fn  = perIterFilename(env, token, delta);
    fp  = fullfile(dataDir, fn);
    if isfile(fp)
        P  = readtable(fp);
        ts = P.tree_size(:);
        it = P.iteration(:);
        pick = round(linspace(1, numel(ts), K));
        chkN = min(max(ts(pick), 1), nNodes);
        lbl  = arrayfun(@(a, b) sprintf('iter %d  (N=%d)', a, b), ...
                        it(pick), chkN, 'UniformOutput', false);
    else
        chkN = round(linspace(max(1, round(nNodes / K)), nNodes, K))';
        lbl  = arrayfun(@(b) sprintf('N=%d', b), chkN, 'UniformOutput', false);
    end
    [chkN, ia] = unique(chkN, 'stable'); lbl = lbl(ia);
end

function fn = perIterFilename(env, token, delta)
    % Mirror writePerIterationCSV naming for run 0 (KinoPaxPlus has no planner token).
    switch token
        case 'KinoPaxPlus'
            fn = sprintf('%s_delta%s_run0.csv', env, delta);
        case {'KPAX', 'PruneKPAX', 'KinoPaxSTAR', 'KinoPaxSTARNoPrune', ...
              'KinoPaxSTARNoPruneNoSpatialHash'}
            fn = sprintf('%s_%s_delta%s_run0.csv', env, token, delta);
        otherwise   % KinoPaxSTARcostprune_capNN
            fn = sprintf('%s_%s_delta%s_run0.csv', env, token, delta);
    end
end

function rgb = val2rgb(v, clim, cmap)
    cmax = max(clim(2), 2);
    t  = log(max(v, 1)) / log(cmax);         % log-normalized 0..1
    t  = min(max(t, 0), 1);
    ci = 1 + round(t * (size(cmap, 1) - 1));
    rgb = cmap(ci, :);
end

function cmap = pickColormap()
    try
        cmap = turbo(256);
    catch
        cmap = parula(256);
    end
end
