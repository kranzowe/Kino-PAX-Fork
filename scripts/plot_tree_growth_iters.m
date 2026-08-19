%% Tree Growth over the First 8 Iterations — 4 configs x 8 iterations, 3D, tiled
% Renders the per-iteration tree dumps written by examples/gpu/tree_growth_dump.cu
% (run via scripts/run_tree_growth_dump.sh) on the zigzag corridor.
%
% Rows are planner configurations, columns are iterations 1..8:
%   ancestor_off  KinoPaxSTARancestor, h_ancestorPrune_ = 0  (== stock KinoPaxSTAR)
%   ancestor_on   KinoPaxSTARancestor, h_ancestorPrune_ = 2  (memoized ancestor chain)
%   KPAX          pure explorer reference
%   KinoPaxPlus   pure optimizer reference (source of the original ancestor pruning)
%
% Every tile is a real 3D axes and ROTATABLE. The camera of all 32 tiles is linked,
% so dragging any one tile turns the whole grid together.
%
% Only 10% of each tree's nodes are drawn, sampled evenly over insertion order so
% the temporal spread survives the subsampling.
%
% Input (in dataDir):
%   {env}_{token}_iter{k}_tree.csv   columns idx,x,y,z,vx,vy,vz,parent,cost
%   meta.csv                         workspace bounds + start/goal
%
% USAGE: cd into the dump directory, then call this script BY NAME, not via run():
%   cd build/Data/Viz/TreeGrowth
%   addpath('<repo>/scripts')
%   plot_tree_growth_iters
% run('<abs path>/plot_tree_growth_iters.m') would cd to the scripts folder first,
% and dataDir below ('' = current folder) would then find nothing.

clear; clc; close all;

%% --- Configuration ---
dataDir = '';   % '' = current directory (run this from Data/Viz/TreeGrowth)
envName = 'zigzag';

% Obstacles: tried relative to the repo root first, then beside the dump.
obstacleCandidates = { ...
    fullfile('..', '..', '..', '..', 'include', 'config', 'obstacles', 'zigzag', 'obstacles.csv'), ...
    fullfile('include', 'config', 'obstacles', 'zigzag', 'obstacles.csv'), ...
    'obstacles.csv'};

tokens       = {'ancestor_off', 'ancestor_on', 'KPAX', 'KinoPaxPlus'};
tokenLabels  = {'STAR (prune off)', 'STAR (ancestor prune)', 'KPAX', 'KinoPaxPlus'};
numIters     = 8;

NODE_FRACTION  = 0.10;    % fraction of nodes drawn per tile
EDGE_FRACTION  = 0.10;    % fraction of parent edges drawn per tile
DRAW_OBSTACLES = true;
OBSTACLE_ALPHA = 0.06;    % low: 5 walls stack along the view axis and alpha compounds
NODE_SIZE      = 9;       % marker area; tiles are small, so bigger than a single-axes plot
VIEW_AZ        = -32;
VIEW_EL        = 26;

%% --- Load meta + obstacles ---
metaPath = fullfile(dataDir, 'meta.csv');
if ~isfile(metaPath)
    error(['meta.csv not found in "%s". Run scripts/run_tree_growth_dump.sh first, ' ...
           'then cd to build/Data/Viz/TreeGrowth.'], pwd);
end
M = table2struct(readtable(metaPath));

obstacles = [];
if DRAW_OBSTACLES
    for c = 1:numel(obstacleCandidates)
        if isfile(obstacleCandidates{c})
            obstacles = readmatrix(obstacleCandidates{c});
            fprintf('Obstacles: %s (%d boxes)\n', obstacleCandidates{c}, size(obstacles, 1));
            break;
        end
    end
    if isempty(obstacles)
        warning('No obstacle CSV found; drawing trees without the environment.');
    end
end

%% --- Load every (config, iteration) tree ---
nTok = numel(tokens);
T    = cell(nTok, numIters);
for ti = 1:nTok
    found = 0;
    for k = 1:numIters
        fp = fullfile(dataDir, sprintf('%s_%s_iter%d_tree.csv', envName, tokens{ti}, k));
        if isfile(fp)
            T{ti, k} = readtable(fp);
            found = found + 1;
        end
    end
    fprintf('  %-22s : %d/%d iterations\n', tokens{ti}, found, numIters);
end

%% --- Figure: nTok rows x numIters columns ---
% Size the figure so each tile comes out roughly square. axis equal letterboxes a
% cube inside a non-square tile, and the leftover vertical slack is what makes the
% row labels look like they are floating between rows.
figW  = min(300 * numIters, 1900);
tileW = figW / numIters;
figH  = tileW * nTok + 70;          % + a band for the layout title
fig = figure('Name', sprintf('Tree Growth (first %d iterations) - %s', numIters, envName), ...
             'Position', [20 20 figW figH], 'Color', 'w');
tl = tiledlayout(fig, nTok, numIters, 'TileSpacing', 'compact', 'Padding', 'compact');

cmap = pickColormap();
axAll = gobjects(nTok * numIters, 1);
a = 0;

for ti = 1:nTok
    % Shared colour scale across the row: node colour is insertion index, and the
    % last iteration spans the widest range. Without this each tile renormalizes
    % and the row stops reading as growth.
    rowMax = 1;
    for k = 1:numIters
        if ~isempty(T{ti, k}), rowMax = max(rowMax, height(T{ti, k})); end
    end

    for k = 1:numIters
        ax = nexttile(tl); hold(ax, 'on');
        a = a + 1; axAll(a) = ax;
        colormap(ax, cmap);

        if ~isempty(obstacles), drawObstacles(ax, obstacles, OBSTACLE_ALPHA); end
        drawWorkspaceCube(ax, M);

        if isempty(T{ti, k})
            tileTxt = sprintf('iter %d  (no data)', k);
            tileCol = [0.6 0.6 0.6];
        else
            drawTree(ax, T{ti, k}, NODE_FRACTION, EDGE_FRACTION, NODE_SIZE);
            clim(ax, [0 rowMax]);
            tileTxt = sprintf('iter %d  (N=%d)', k, height(T{ti, k}));
            tileCol = [0 0 0];
        end

        % The row name goes in the leftmost tile's title. ylabel() would sit beside
        % the 3-D y-axis, which reads as labelling the gap between rows, not the row.
        if k == 1
            title(ax, {tokenLabels{ti}, tileTxt}, 'FontSize', 8, 'Color', tileCol);
        else
            title(ax, tileTxt, 'FontSize', 7.5, 'Color', tileCol);
        end

        drawStartGoal(ax, M);
        finishAxes(ax, M, VIEW_AZ, VIEW_EL);
    end
end

title(tl, {sprintf('Tree growth over the first %d iterations \x2014 %s', numIters, envName), ...
           sprintf('%.0f%% of nodes drawn, colour = insertion order', 100 * NODE_FRACTION)}, ...
      'FontWeight', 'bold', 'FontSize', 11, 'Interpreter', 'none');

% Link all tile cameras so dragging one rotates the whole grid. The handle must be
% kept alive or MATLAB garbage-collects the link and rotation silently desyncs.
axAll = axAll(isgraphics(axAll));
fig.UserData.cameraLink = linkprop(axAll, ...
    {'CameraPosition', 'CameraTarget', 'CameraUpVector', 'CameraViewAngle'});
rotate3d(fig, 'on');

fprintf('\nDone. Drag any tile to rotate all %d together.\n', numel(axAll));

%% ====================== helper functions ======================

function drawTree(ax, Tb, nodeFrac, edgeFrac, nodeSize)
    % Parent edges (NaN-separated segments) plus nodes coloured by insertion order.
    % Both are subsampled by FRACTION, evenly spaced over insertion order rather
    % than a head/tail slice, so early and late structure both survive.
    n = height(Tb);
    if n == 0, return; end

    p = Tb.parent;                 % 0-indexed parent, -1 at the root
    child = find(p >= 0);
    if edgeFrac > 0 && ~isempty(child)
        nEdge = max(1, round(edgeFrac * numel(child)));
        if nEdge < numel(child)
            child = child(unique(round(linspace(1, numel(child), nEdge))));
        end
        pr = p(child) + 1;         % parent row (idx is 0-based, rows are in idx order)
        X = [Tb.x(child)'; Tb.x(pr)'; nan(1, numel(child))];
        Y = [Tb.y(child)'; Tb.y(pr)'; nan(1, numel(child))];
        Z = [Tb.z(child)'; Tb.z(pr)'; nan(1, numel(child))];
        plot3(ax, X(:), Y(:), Z(:), '-', 'Color', [0.72 0.72 0.75], 'LineWidth', 0.3);
    end

    nNode = max(1, round(nodeFrac * n));
    ridx  = unique(round(linspace(1, n, min(nNode, n))));
    scatter3(ax, Tb.x(ridx), Tb.y(ridx), Tb.z(ridx), nodeSize, Tb.idx(ridx), 'filled', ...
             'MarkerFaceAlpha', 0.9);
end

function drawObstacles(ax, obstacles, alpha)
    % All boxes as ONE patch object. visualize_tree_growth.m issues one patch per
    % box, which is fine for a single axes but costs 20 x 32 = 640 transparent
    % objects across this grid and makes both rotation and export crawl.
    if isempty(obstacles), return; end
    nB = size(obstacles, 1);
    faces1 = [1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8; 1 2 3 4; 5 6 7 8];
    V = zeros(8 * nB, 3);
    F = zeros(6 * nB, 4);
    for j = 1:nB
        o = obstacles(j, :);   % xmin ymin zmin xmax ymax zmax
        V(8 * (j - 1) + (1:8), :) = ...
            [o(1) o(2) o(3); o(4) o(2) o(3); o(4) o(5) o(3); o(1) o(5) o(3); ...
             o(1) o(2) o(6); o(4) o(2) o(6); o(4) o(5) o(6); o(1) o(5) o(6)];
        F(6 * (j - 1) + (1:6), :) = faces1 + 8 * (j - 1);
    end
    patch(ax, 'Vertices', V, 'Faces', F, 'FaceColor', [0.45 0.5 0.58], ...
          'EdgeColor', [0.3 0.3 0.3], 'FaceAlpha', alpha, 'EdgeAlpha', alpha, ...
          'HandleVisibility', 'off');
end

function drawWorkspaceCube(ax, M)
    % 12-edge wireframe of the workspace bounds, for depth cueing.
    lo = M.W_MIN; hi = M.W_MAX;
    V = [lo lo lo; hi lo lo; hi hi lo; lo hi lo; lo lo hi; hi lo hi; hi hi hi; lo hi hi];
    E = [1 2; 2 3; 3 4; 4 1; 5 6; 6 7; 7 8; 8 5; 1 5; 2 6; 3 7; 4 8];
    % One NaN-separated line object rather than 12, for the same reason as drawObstacles.
    X = [V(E(:, 1), 1)'; V(E(:, 2), 1)'; nan(1, size(E, 1))];
    Y = [V(E(:, 1), 2)'; V(E(:, 2), 2)'; nan(1, size(E, 1))];
    Z = [V(E(:, 1), 3)'; V(E(:, 2), 3)'; nan(1, size(E, 1))];
    plot3(ax, X(:), Y(:), Z(:), '-', 'Color', [0.85 0.85 0.85], ...
          'LineWidth', 0.4, 'HandleVisibility', 'off');
end

function drawStartGoal(ax, M)
    plot3(ax, M.start_x, M.start_y, M.start_z, 'o', 'MarkerSize', 5, ...
          'MarkerFaceColor', [0.15 0.70 0.25], 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
    plot3(ax, M.goal_x, M.goal_y, M.goal_z, 'p', 'MarkerSize', 8, ...
          'MarkerFaceColor', [0.85 0.15 0.15], 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
end

function finishAxes(ax, M, az, el)
    axis(ax, 'equal');
    xlim(ax, [M.W_MIN M.W_MAX]); ylim(ax, [M.W_MIN M.W_MAX]); zlim(ax, [M.W_MIN M.W_MAX]);
    grid(ax, 'on'); box(ax, 'on');
    view(ax, az, el);
    set(ax, 'FontSize', 7, 'XTickLabel', [], 'YTickLabel', [], 'ZTickLabel', []);
end

function cmap = pickColormap()
    try
        cmap = turbo(256);
    catch
        cmap = parula(256);
    end
end
