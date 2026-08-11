%% satellite_tree_growth.m
% Animated KinoPaxSTAR search-tree growth for the satellite flag-capture scenario,
% from the CSVs written by the `satellite_treegrowth` executable
% (Data/SatelliteTree/). Nodes/edges are revealed in birth-iteration order; nodes
% are colored by cost-to-come (DV^2 + safety). The final solution path (if the run
% reached the goal) is overlaid in bold.
%
% EDIT `dataDir` to point at the copied output folder, then run.

clear; close all;

%% ------------------------- config -------------------------
thisDir     = fileparts(mfilename('fullpath'));
dataDir     = fullfile(thisDir, '..', 'build', 'Data', 'SatelliteTree');  % <-- EDIT if needed
exportVideo = true;
videoFile   = fullfile(dataDir, 'tree_growth.mp4');
frameRate   = 20;
maxFrames   = 150;    % cap animation frames (iterations are subsampled to this)

%% ------------------------- load ---------------------------
meta = readtable(fullfile(dataDir, 'meta.csv'));
T    = readtable(fullfile(dataDir, 'tree.csv'));       % idx,x,y,vx,vy,parent,cost,iter
obs  = readtable(fullfile(dataDir, 'obstacles.csv'));  % xmin,ymin,xmax,ymax
sol  = [];
solFile = fullfile(dataDir, 'solution.csv');
if isfile(solFile), s = readtable(solFile); if height(s) > 0, sol = [s.x s.y]; end, end

flag  = [meta.flag_x, meta.flag_y];
start = [meta.start_x, meta.start_y];
Rkeep = meta.R_KEEPOUT;
capR  = meta.GOAL_THRESH;

% Edges child->parent. Reveal order = node insertion order (idx), which is the tree's
% true growth order since nodes are only ever appended (same convention as
% visualize_tree_growth.m). NOTE: the `iter` column is filled by the planner only for
% goal nodes, so idx -- not iter -- drives the growth animation.
hasP  = T.parent >= 0;
ci    = find(hasP);                             % 1-based child rows (child idx = ci-1)
pr    = T.parent(ci) + 1;                       % 0-based parent -> 1-based row
ex    = [T.x(ci)'; T.x(pr)'; nan(1, numel(ci))];
ey    = [T.y(ci)'; T.y(pr)'; nan(1, numel(ci))];

nNodes    = height(T);
revCounts = unique(round(linspace(1, nNodes, min(nNodes, maxFrames))));

% Robust color scale for cost-to-come.
cmax = prctile(T.cost(isfinite(T.cost)), 98);
if ~(cmax > 0), cmax = 1; end

%% ------------------------- view ---------------------------
thc  = linspace(0, 2*pi, 60);
allx = [T.x; obs.xmin; obs.xmax; flag(1); start(1)];
ally = [T.y; obs.ymin; obs.ymax; flag(2); start(2)];
pad  = 150;
cx   = 0.5*(min(allx)+max(allx)); cy = 0.5*(min(ally)+max(ally));
half = 0.5*max(max(allx)-min(allx), max(ally)-min(ally)) + pad;

fig = figure('Color', 'w', 'Position', [100 100 860 860]);
ax  = axes(fig); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
xlim(ax, [cx-half, cx+half]); ylim(ax, [cy-half, cy+half]);
daspect(ax, [1 1 1]);   % equal units; square limits keep the box square (see anim script)
colormap(ax, parula); caxis(ax, [0 cmax]);
cb = colorbar(ax); cb.Label.String = 'cost-to-come  (DV^2 + safety)';
xlabel(ax, 'Radial  x  [m]'); ylabel(ax, 'In-track  y  [m]');
title(ax, 'KinoPaxSTAR search-tree growth');

% Static scene: obstacle keep-out boxes + circles, start, flag, capture ring.
for k = 1:height(obs)
    ox = 0.5*(obs.xmin(k)+obs.xmax(k)); oy = 0.5*(obs.ymin(k)+obs.ymax(k));
    rectangle(ax, 'Position', [obs.xmin(k) obs.ymin(k) obs.xmax(k)-obs.xmin(k) obs.ymax(k)-obs.ymin(k)], ...
              'EdgeColor', [0.85 0.3 0.3], 'FaceColor', 'none');
    plot(ax, ox + Rkeep*cos(thc), oy + Rkeep*sin(thc), '-', 'Color', [0.85 0.3 0.3]);
end
plot(ax, start(1), start(2), 's', 'MarkerSize', 11, 'MarkerFaceColor', [0.1 0.6 0.2], 'MarkerEdgeColor', 'k');
plot(ax, flag(1), flag(2), 'p', 'MarkerSize', 20, 'MarkerFaceColor', [0.95 0.80 0.10], 'MarkerEdgeColor', 'k');
plot(ax, flag(1) + capR*cos(thc), flag(2) + capR*sin(thc), ':', 'Color', [0.5 0.5 0.5]);

% Dynamic handles.
hEdges = plot(ax, nan, nan, '-', 'Color', [0.72 0.72 0.72], 'LineWidth', 0.4);
hNodes = scatter(ax, nan, nan, 8, 'filled');
hSol   = plot(ax, nan, nan, '-', 'Color', [0.85 0.1 0.1], 'LineWidth', 2.2);
hTxt   = text(ax, cx-half+40, cy+half-60, '', 'FontSize', 11, 'BackgroundColor', [1 1 1], 'EdgeColor', [0.8 0.8 0.8]);

%% ------------------------- video --------------------------
useGif = false; vw = []; gifStarted = false;
if exportVideo
    try
        vw = VideoWriter(videoFile, 'MPEG-4'); vw.FrameRate = frameRate; open(vw);
    catch
        videoFile = strrep(videoFile, '.mp4', '.gif'); useGif = true;
    end
end

%% ------------------------- animate ------------------------
for fi = 1:numel(revCounts)
    nRev = revCounts(fi);
    em   = ci <= nRev;                          % edge shown once its child node is revealed
    set(hEdges, 'XData', reshape(ex(:, em), [], 1), 'YData', reshape(ey(:, em), [], 1));
    set(hNodes, 'XData', T.x(1:nRev), 'YData', T.y(1:nRev), 'CData', min(T.cost(1:nRev), cmax));
    set(hTxt, 'String', sprintf('nodes %d / %d', nRev, nNodes));
    drawnow;
    if exportVideo
        fr = getframe(fig);
        if useGif
            [A, map] = rgb2ind(frame2im(fr), 256);
            if ~gifStarted
                imwrite(A, map, videoFile, 'gif', 'LoopCount', Inf, 'DelayTime', 1/frameRate); gifStarted = true;
            else
                imwrite(A, map, videoFile, 'gif', 'WriteMode', 'append', 'DelayTime', 1/frameRate);
            end
        else
            writeVideo(vw, fr);
        end
    end
end

% Overlay the solution path (real tree nodes only; the flag is never appended).
nIters = meta.iterations;
if ~isempty(sol)
    set(hSol, 'XData', sol(:,1), 'YData', sol(:,2));
    set(hTxt, 'String', sprintf('done: %d nodes, %d iters  |  solution: %d waypoints', nNodes, nIters, size(sol,1)));
else
    set(hTxt, 'String', sprintf('done: %d nodes, %d iters  |  no goal reached', nNodes, nIters));
end
drawnow;
if exportVideo
    for r = 1:frameRate   % hold the final frame ~1 s
        fr = getframe(fig);
        if useGif
            [A, map] = rgb2ind(frame2im(fr), 256);
            imwrite(A, map, videoFile, 'gif', 'WriteMode', 'append', 'DelayTime', 1/frameRate);
        else
            writeVideo(vw, fr);
        end
    end
    if ~useGif, close(vw); end
end
fprintf('Done. Output: %s\n', videoFile);
