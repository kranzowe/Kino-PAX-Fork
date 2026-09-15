%% Tree Checkpoint — top-down, 3 panels (one per planner) at one chosen time
% Renders the wall-clock-checkpoint tree + solution-trajectory dumps written by
% examples/gpu/tree_checkpoint_dump.cu (run via scripts/run_tree_checkpoint_dump.sh)
% on the zigzag corridor, Model 1, V2's canonical "large"/coarse discretization.
%
% Panels (left to right): KPAX, KinoPaxPlus, KinoPaxSTARTrue -- each showing that
% planner's tree at the SAME chosen checkpoint, top-down (X-Y), with its solution
% trajectory overlaid (if one existed yet at that checkpoint).
%
% Input (in dataDir):
%   {env}_{token}_t{ms}ms_tree.csv   columns idx,x,y,z,vx,vy,vz,parent,cost
%   {env}_{token}_t{ms}ms_traj.csv   columns step,x,y,z,vx,vy,vz,cost (0 rows if unsolved yet)
%   meta.csv                         workspace bounds + start/goal
%
% USAGE: cd into the dump directory, then call this script BY NAME, not via run():
%   cd build/Data/Viz/TreeCheckpoints
%   addpath('<repo>/scripts')
%   plot_tree_checkpoint
% run('<abs path>/plot_tree_checkpoint.m') would cd to the scripts folder first,
% and dataDir below ('' = current folder) would then find nothing.

clear; clc; close all;

%% --- Configuration ---
dataDir = '';   % '' = current directory (run this from Data/Viz/TreeCheckpoints)
envName = 'zigzag';

% Pick ONE checkpoint to render (uncomment one):
checkpointMs = 100;
% checkpointMs = 50;
% checkpointMs = 150;
% checkpointMs = 1000;
% checkpointMs = 4000;
validCheckpoints = [50 100 150 1000 4000];
assert(ismember(checkpointMs, validCheckpoints), ...
       'checkpointMs must be one of %s', mat2str(validCheckpoints));

% Obstacles: tried relative to the repo root first, then beside the dump.
obstacleCandidates = { ...
    fullfile('..', '..', '..', '..', 'include', 'config', 'obstacles', 'zigzag', 'obstacles.csv'), ...
    fullfile('include', 'config', 'obstacles', 'zigzag', 'obstacles.csv'), ...
    'obstacles.csv'};

tokens      = {'KPAX', 'KinoPaxPlus', 'KinoPaxSTARTrue'};
tokenLabels = {'KPAX', 'KinoPaxPlus', 'KinoPaxSTARTrue'};

NODE_FRACTION  = 0.15;    % fraction of nodes drawn per panel
EDGE_FRACTION  = 0.15;    % fraction of parent edges drawn per panel
DRAW_OBSTACLES = true;
OBSTACLE_ALPHA = 0.35;    % top-down = obstacles don't stack along the view axis like 3D does
NODE_SIZE      = 6;
TRAJ_LINEWIDTH = 2.2;
TRAJ_COLOR     = [0.10 0.35 0.95];

%% --- Load meta + obstacles ---
metaPath = fullfile(dataDir, 'meta.csv');
if ~isfile(metaPath)
    error(['meta.csv not found in "%s". Run scripts/run_tree_checkpoint_dump.sh first, ' ...
           'then cd to build/Data/Viz/TreeCheckpoints.'], pwd);
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

%% --- Load each planner's tree + trajectory at the chosen checkpoint ---
nTok = numel(tokens);
Tree = cell(nTok, 1);
Traj = cell(nTok, 1);
for ti = 1:nTok
    treeFp = fullfile(dataDir, sprintf('%s_%s_t%dms_tree.csv', envName, tokens{ti}, checkpointMs));
    trajFp = fullfile(dataDir, sprintf('%s_%s_t%dms_traj.csv', envName, tokens{ti}, checkpointMs));
    if isfile(treeFp)
        Tree{ti} = readtable(treeFp);
    else
        warning('Missing tree file: %s', treeFp);
    end
    if isfile(trajFp)
        Traj{ti} = readtable(trajFp);   % may legitimately have 0 rows (no solution yet)
    else
        warning('Missing trajectory file: %s', trajFp);
    end
end

%% --- Figure: 1 row x 3 columns, top-down ---
figW = 1500; figH = 560;
fig = figure('Name', sprintf('Tree Checkpoint @ %dms - %s', checkpointMs, envName), ...
             'Position', [20 20 figW figH], 'Color', 'w');
tl = tiledlayout(fig, 1, nTok, 'TileSpacing', 'compact', 'Padding', 'compact');

cmap = pickColormap();

for ti = 1:nTok
    ax = nexttile(tl); hold(ax, 'on');
    colormap(ax, cmap);

    if ~isempty(obstacles), drawObstacles2D(ax, obstacles, OBSTACLE_ALPHA); end
    drawWorkspaceSquare(ax, M);

    if isempty(Tree{ti})
        tileTxt = 'no data';
        tileCol = [0.6 0.6 0.6];
    else
        drawTree2D(ax, Tree{ti}, NODE_FRACTION, EDGE_FRACTION, NODE_SIZE);
        nSolved = ~isempty(Traj{ti}) && height(Traj{ti}) > 0;
        if nSolved
            drawTrajectory2D(ax, Traj{ti}, TRAJ_COLOR, TRAJ_LINEWIDTH);
            tileTxt = sprintf('N=%d nodes | SOLVED, cost=%.4f', height(Tree{ti}), Traj{ti}.cost(end));
            tileCol = [0.05 0.45 0.10];
        else
            tileTxt = sprintf('N=%d nodes | no solution yet', height(Tree{ti}));
            tileCol = [0.6 0.15 0.15];
        end
    end

    title(ax, {tokenLabels{ti}, tileTxt}, 'FontSize', 9, 'Color', tileCol);
    drawStartGoal2D(ax, M);
    finishAxes2D(ax, M);
end

title(tl, sprintf('Tree + solution trajectory @ %dms \x2014 %s (Model 1, coarse/large)', ...
      checkpointMs, envName), 'FontWeight', 'bold', 'FontSize', 12, 'Interpreter', 'none');

fprintf('\nDone. Rendered checkpoint t=%dms for %d planners.\n', checkpointMs, nTok);

%% ====================== helper functions ======================

function drawTree2D(ax, Tb, nodeFrac, edgeFrac, nodeSize)
    % Top-down (X-Y) version of plot_tree_growth_iters.m's drawTree: parent edges
    % (NaN-separated segments) plus nodes coloured by insertion order, both
    % subsampled evenly over insertion order so early/late structure both survive.
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
        plot(ax, X(:), Y(:), '-', 'Color', [0.72 0.72 0.75], 'LineWidth', 0.3);
    end

    nNode = max(1, round(nodeFrac * n));
    ridx  = unique(round(linspace(1, n, min(nNode, n))));
    scatter(ax, Tb.x(ridx), Tb.y(ridx), nodeSize, Tb.idx(ridx), 'filled', ...
            'MarkerFaceAlpha', 0.8);
end

function drawTrajectory2D(ax, Traj, color, lw)
    plot(ax, Traj.x, Traj.y, '-', 'Color', color, 'LineWidth', lw, 'HandleVisibility', 'off');
end

function drawObstacles2D(ax, obstacles, alpha)
    % Top-down X-Y footprint of every box, ignoring z-extent entirely (a projection,
    % not a slice) -- unlike the 3D scripts' patch-based drawObstacles, no existing
    % script has a top-down projection to adapt, so this uses rectangle() directly,
    % matching view_environment.m's established use of rectangle() for 2D drawing.
    nB = size(obstacles, 1);
    for j = 1:nB
        o = obstacles(j, :);   % xmin ymin zmin xmax ymax zmax
        rectangle(ax, 'Position', [o(1) o(2) o(4) - o(1) o(5) - o(2)], ...
                  'FaceColor', [0.45 0.5 0.58 alpha], 'EdgeColor', [0.3 0.3 0.3 alpha], ...
                  'LineWidth', 0.5);
    end
end

function drawWorkspaceSquare(ax, M)
    lo = M.W_MIN; hi = M.W_MAX;
    rectangle(ax, 'Position', [lo lo hi - lo hi - lo], ...
              'EdgeColor', [0.85 0.85 0.85], 'LineWidth', 0.6, 'FaceColor', 'none');
end

function drawStartGoal2D(ax, M)
    plot(ax, M.start_x, M.start_y, 'o', 'MarkerSize', 6, ...
         'MarkerFaceColor', [0.15 0.70 0.25], 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
    plot(ax, M.goal_x, M.goal_y, 'p', 'MarkerSize', 10, ...
         'MarkerFaceColor', [0.85 0.15 0.15], 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
end

function finishAxes2D(ax, M)
    axis(ax, 'equal');
    xlim(ax, [M.W_MIN M.W_MAX]); ylim(ax, [M.W_MIN M.W_MAX]);
    grid(ax, 'on'); box(ax, 'on');
    view(ax, 2);   % top-down
    set(ax, 'FontSize', 8);
    xlabel(ax, 'x'); ylabel(ax, 'y');
end

function cmap = pickColormap()
    try
        cmap = turbo(256);
    catch
        cmap = parula(256);
    end
end
