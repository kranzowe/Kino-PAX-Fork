%% Environment Viewer — render an obstacle CSV as 3D boxes
% Quick visual check that an obstacle set looks the way it was meant to, before
% burning an hour of benchmark time on it.
%
% CSV format (matches readObstaclesFromCSV / isBroadPhaseValid in src/helper,
% src/collisionCheck): one axis-aligned box per row,
%   xmin,ymin,zmin,xmax,ymax,zmax
% in workspace coordinates — [0,1]^3 for MODEL 1.
%
% USAGE: from the repo root,
%   addpath('scripts'); view_environment
% or point it at a different file / start / goal by editing the config below.
%
% Produces two figures:
%   1. 3D view of the boxes with the start (green) and goal (red) markers, plus
%      the straight start->goal line for scale.
%   2. Per-wall face slices: for each distinct wall (grouped by y extent), a 2D
%      x-z map of what is solid and what is open. This is the one that actually
%      tells you whether the narrow passages are where you think they are.

clear; clc; close all;

%% --- Configuration ---
obstacleFile = fullfile('include', 'config', 'obstacles', 'zigzag', 'obstacles.csv');
envName      = 'zigzag';

% Start / goal in workspace coords — keep in sync with the benchmark's main().
% kinopaxstar_cost_tuning_sweep.cu uses W_MIN + frac*W_SIZE with W_MIN=0, W_SIZE=1.
startPt = [0.10 0.08 0.05];
goalPt  = [0.80 0.95 0.90];

wsMin = 0; wsMax = 1;      % W_MIN / W_MAX from include/config/config.h
faceAlpha = 0.28;

%% --- Load ---
if ~isfile(obstacleFile)
    error('obstacle file not found: %s (run this from the repo root)', obstacleFile);
end
O = readmatrix(obstacleFile);
if size(O, 2) ~= 6
    error('expected 6 columns (xmin,ymin,zmin,xmax,ymax,zmax), got %d', size(O, 2));
end
nObs = size(O, 1);
fprintf('Loaded %d obstacles from %s\n', nObs, obstacleFile);

% Report anything degenerate or out of bounds — a silent typo here shows up much
% later as an unexplainable planner failure.
degenerate = find(any(O(:, 1:3) >= O(:, 4:6), 2));
if ~isempty(degenerate)
    warning('%d box(es) have min >= max: rows %s', numel(degenerate), mat2str(degenerate'));
end
oob = find(any(O(:, 1:3) < wsMin - 1e-6, 2) | any(O(:, 4:6) > wsMax + 1e-6, 2));
if ~isempty(oob)
    warning('%d box(es) extend outside [%g,%g]^3: rows %s', numel(oob), wsMin, wsMax, mat2str(oob'));
end

% Start / goal must be in free space or the benchmark cannot even begin.
checkFree(O, startPt, 'start');
checkFree(O, goalPt,  'goal');

% Group boxes by their (ymin,ymax) extent — each group is one "wall". Used for
% per-wall colouring below and for the face slices in figure 2.
[yExt, ~, grp] = unique(O(:, [2 5]), 'rows');
nWall = size(yExt, 1);

%% ---------- FIGURE 1: 3D view ----------
figure('Name', sprintf('%s - 3D', envName), 'Position', [60 60 980 780]);
hold on;
% One hue per wall: five translucent walls in a single colour overlap into an
% unreadable mush, and the whole point of this figure is telling them apart.
wallColors = parula(max(nWall, 2));
for i = 1:nObs
    drawBox(O(i, 1:3), O(i, 4:6), wallColors(grp(i), :) * 0.85, faceAlpha);
end
plot3(startPt(1), startPt(2), startPt(3), 'o', 'MarkerSize', 12, ...
      'MarkerFaceColor', [0.15 0.70 0.25], 'MarkerEdgeColor', 'k', 'DisplayName', 'start');
plot3(goalPt(1), goalPt(2), goalPt(3), 'p', 'MarkerSize', 18, ...
      'MarkerFaceColor', [0.85 0.15 0.15], 'MarkerEdgeColor', 'k', 'DisplayName', 'goal');
plot3([startPt(1) goalPt(1)], [startPt(2) goalPt(2)], [startPt(3) goalPt(3)], ...
      'k--', 'LineWidth', 1, 'DisplayName', 'straight line');
axis equal; grid on; box on;
xlim([wsMin wsMax]); ylim([wsMin wsMax]); zlim([wsMin wsMax]);
xlabel('x'); ylabel('y'); zlabel('z');
view(-32, 26); camlight headlight; lighting gouraud; rotate3d on;
legend('Location', 'northeast');
title(sprintf('%s \x2014 %d obstacles', envName, nObs), 'Interpreter', 'none', 'FontWeight', 'bold');

%% ---------- FIGURE 2: per-wall face slices ----------
% Each wall's face drawn head-on in x-z. Openings are whatever the boxes do not
% cover — this is the view that shows where the narrow passages actually are.
if nWall >= 1 && nWall <= 12
    nCol = min(nWall, 3);
    nRow = ceil(nWall / nCol);
    figure('Name', sprintf('%s - wall faces', envName), 'Position', [90 90 420 * nCol 380 * nRow]);
    for w = 1:nWall
        subplot(nRow, nCol, w); hold on;
        wall = O(grp == w, :);
        for i = 1:size(wall, 1)
            rectangle('Position', [wall(i, 1), wall(i, 3), ...
                                   wall(i, 4) - wall(i, 1), wall(i, 6) - wall(i, 3)], ...
                      'FaceColor', [0.35 0.45 0.60], 'EdgeColor', [0.15 0.2 0.3]);
        end
        openFrac = 1 - solidFraction(wall);
        axis equal; xlim([wsMin wsMax]); ylim([wsMin wsMax]); box on;
        xlabel('x'); ylabel('z');
        title(sprintf('y \x2208 [%.2f, %.2f] \x2014 %.1f%% open', ...
              yExt(w, 1), yExt(w, 2), 100 * openFrac), 'FontSize', 9);
    end
    sgtitle(sprintf('%s \x2014 wall faces (blue = solid, white = passable)', envName), ...
            'Interpreter', 'none', 'FontWeight', 'bold');
else
    fprintf('Skipping face slices: %d distinct y-extents (expected 1..12).\n', nWall);
end

fprintf('Done. Rotate figure 1 to inspect; figure 2 shows each wall face head-on.\n');

%% ====================== helper functions ======================

function drawBox(lo, hi, color, alpha)
    % Six quads of an axis-aligned box.
    x = [lo(1) hi(1)]; y = [lo(2) hi(2)]; z = [lo(3) hi(3)];
    V = [x(1) y(1) z(1); x(2) y(1) z(1); x(2) y(2) z(1); x(1) y(2) z(1); ...
         x(1) y(1) z(2); x(2) y(1) z(2); x(2) y(2) z(2); x(1) y(2) z(2)];
    F = [1 2 3 4; 5 6 7 8; 1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8];
    patch('Vertices', V, 'Faces', F, 'FaceColor', color, 'FaceAlpha', alpha, ...
          'EdgeColor', [0.2 0.25 0.35], 'LineWidth', 0.4, 'HandleVisibility', 'off');
end

function checkFree(O, p, name)
    % A point is blocked if it lies inside any box (inclusive bounds).
    hit = find(all(O(:, 1:3) <= p, 2) & all(p <= O(:, 4:6), 2));
    if isempty(hit)
        fprintf('  %-5s [%.2f %.2f %.2f] : FREE\n', name, p(1), p(2), p(3));
    else
        warning('%s [%.2f %.2f %.2f] is INSIDE obstacle row(s) %s', ...
                name, p(1), p(2), p(3), mat2str(hit'));
    end
end

function f = solidFraction(wall)
    % Fraction of the unit x-z face covered by this wall's boxes, by rasterizing.
    % Sampled rather than analytic, which is plenty for a visual sanity check.
    n = 400;
    c = (0.5:n) / n;
    [X, Z] = meshgrid(c, c);
    solid = false(size(X));
    for i = 1:size(wall, 1)
        solid = solid | (X >= wall(i, 1) & X <= wall(i, 4) & Z >= wall(i, 3) & Z <= wall(i, 6));
    end
    f = mean(solid(:));
end
