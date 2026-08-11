%% satellite_tree_growth.m
% Animated KinoPaxSTAR search-tree growth for the satellite flag-capture scenario,
% from the CSVs written by the `satellite_treegrowth` executable (Data/SatelliteTree/).
%
% Shows only the FIRST `numIters` iterations of the algorithm, revealed one iteration
% at a time (nodes colored by cost-to-come). Any path that reaches the goal ball within
% those iterations is highlighted in bold magenta, traced back to the start.
%
% EDIT `dataDir` to point at the copied output folder, then run.

clear; close all;

%% ------------------------- config -------------------------
thisDir     = fileparts(mfilename('fullpath'));
dataDir     = fullfile(thisDir, '..', 'build', 'Data', 'SatelliteTree');  % <-- EDIT if needed
numIters    = 10;      % show only the first N algorithm iterations
exportVideo = true;
videoFile   = fullfile(dataDir, 'tree_growth.mp4');
frameRate   = 20;
holdFrames  = 12;      % video frames to hold on each iteration (pacing)

%% ------------------------- load ---------------------------
meta = readtable(fullfile(dataDir, 'meta.csv'));
T    = readtable(fullfile(dataDir, 'tree.csv'));       % idx,x,y,vx,vy,parent,cost,iter (row i = idx i-1)
obs  = readtable(fullfile(dataDir, 'obstacles.csv'));  % xmin,ymin,xmax,ymax

flag  = [meta.flag_x, meta.flag_y];
start = [meta.start_x, meta.start_y];
Rkeep = meta.R_KEEPOUT;
capR  = meta.GOAL_THRESH;

% Edges child->parent, tagged with the child's birth iteration.
hasP  = T.parent >= 0;
ci    = find(hasP);
pr    = T.parent(ci) + 1;                       % 0-based parent idx -> 1-based row
ex    = [T.x(ci)'; T.x(pr)'; nan(1, numel(ci))];
ey    = [T.y(ci)'; T.y(pr)'; nan(1, numel(ci))];
eIter = T.iter(ci);

% Goal-reaching paths: nodes within the capture ball, born within numIters, traced to
% root. (The flag itself is never a node -- these are real trajectories that reached it.)
d2flag = hypot(T.x - flag(1), T.y - flag(2));
gRows  = find(d2flag < capR & T.iter >= 1 & T.iter <= numIters);
goalPaths = cell(numel(gRows), 1);
goalIter  = zeros(numel(gRows), 1);
for m = 1:numel(gRows)
    row = gRows(m); goalIter(m) = T.iter(row);
    px = T.x(row); py = T.y(row);
    while true
        p = T.parent(row);
        if p < 0, break; end
        row = p + 1; px(end+1) = T.x(row); py(end+1) = T.y(row); %#ok<AGROW>
    end
    goalPaths{m} = [px(:), py(:)];
end

% Color scale from the visible (first-numIters) nodes.
vis  = T.iter <= numIters;
cmax = prctile(T.cost(vis & isfinite(T.cost)), 98);
if ~(cmax > 0), cmax = 1; end

%% ------------------------- view ---------------------------
thc  = linspace(0, 2*pi, 60);
allx = [T.x(vis); obs.xmin; obs.xmax; flag(1); start(1)];
ally = [T.y(vis); obs.ymin; obs.ymax; flag(2); start(2)];
pad  = 150;
cx   = 0.5*(min(allx)+max(allx)); cy = 0.5*(min(ally)+max(ally));
half = 0.5*max(max(allx)-min(allx), max(ally)-min(ally)) + pad;

fig = figure('Color', 'w', 'Position', [100 100 860 860]);
ax  = axes(fig); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
xlim(ax, [cx-half, cx+half]); ylim(ax, [cy-half, cy+half]);
daspect(ax, [1 1 1]);   % equal units; square limits keep the box square
colormap(ax, parula); caxis(ax, [0 cmax]);
cb = colorbar(ax); cb.Label.String = 'cost-to-come  (DV^2 + safety)';
xlabel(ax, 'Radial  x  [m]'); ylabel(ax, 'In-track  y  [m]');
title(ax, sprintf('KinoPaxSTAR tree growth (first %d iterations)', numIters));

% Static scene: obstacle keep-out boxes + circles, start, flag, capture ring.
for k = 1:height(obs)
    ox = 0.5*(obs.xmin(k)+obs.xmax(k)); oy = 0.5*(obs.ymin(k)+obs.ymax(k));
    rectangle('Position', [obs.xmin(k) obs.ymin(k) obs.xmax(k)-obs.xmin(k) obs.ymax(k)-obs.ymin(k)], ...
              'EdgeColor', [0.85 0.3 0.3], 'FaceColor', 'none', 'Parent', ax);
    plot(ax, ox + Rkeep*cos(thc), oy + Rkeep*sin(thc), '-', 'Color', [0.85 0.3 0.3]);
end
plot(ax, start(1), start(2), 's', 'MarkerSize', 11, 'MarkerFaceColor', [0.1 0.6 0.2], 'MarkerEdgeColor', 'k');
plot(ax, flag(1), flag(2), 'p', 'MarkerSize', 20, 'MarkerFaceColor', [0.95 0.80 0.10], 'MarkerEdgeColor', 'k');
plot(ax, flag(1) + capR*cos(thc), flag(2) + capR*sin(thc), ':', 'Color', [0.5 0.5 0.5]);

% Dynamic handles.
hEdges = plot(ax, nan, nan, '-', 'Color', [0.72 0.72 0.72], 'LineWidth', 0.4);
hNodes = scatter(ax, nan, nan, 10, 'filled');
hGoal  = plot(ax, nan, nan, '-', 'Color', [0.90 0.15 0.60], 'LineWidth', 2.6);   % goal-reaching paths
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
for f = 0:numIters
    em = eIter <= f;
    set(hEdges, 'XData', reshape(ex(:, em), [], 1), 'YData', reshape(ey(:, em), [], 1));
    nm = T.iter <= f;
    set(hNodes, 'XData', T.x(nm), 'YData', T.y(nm), 'CData', min(T.cost(nm), cmax));

    % goal-reaching paths discovered so far
    gx = []; gy = [];
    for m = 1:numel(goalPaths)
        if goalIter(m) <= f
            gx = [gx; goalPaths{m}(:,1); nan]; %#ok<AGROW>
            gy = [gy; goalPaths{m}(:,2); nan]; %#ok<AGROW>
        end
    end
    set(hGoal, 'XData', gx, 'YData', gy);

    set(hTxt, 'String', sprintf('iteration %d / %d    nodes %d    goal-reaching paths %d', ...
                                f, numIters, nnz(nm), sum(goalIter <= f)));

    for h = 1:holdFrames
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
end
if exportVideo && ~useGif, close(vw); end

nGoal = numel(goalPaths);
fprintf('Done. First %d iterations: %d nodes shown, %d goal-reaching path(s). Output: %s\n', ...
        numIters, nnz(T.iter <= numIters), nGoal, videoFile);
