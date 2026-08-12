%% satellite_pursuit_anim.m
% Animate the KinoPaxSTARcostprune 2D satellite flag-capture demo from the CSVs written
% by the `satellitecontroller` executable (Data/SatellitePursuit/). Nothing is computed
% here -- the GPU run produces the trajectories; this just renders them.
%
% Shows: flag at the origin (+ capture radius), the satellite with its flown trail, the
% current cycle's smooth nominal plan (dashed) plus an example noisy open-loop fly-out of that
% whole plan (faded), and the defenders as markers with keep-out circles and fading trails.
% Curves update on each replan. Optionally exports a video.
%
% EDIT `dataDir` below to point at the copied output folder, then run.

clear; close all;

%% ------------------------- config -------------------------
dataDir     = '';  % <-- EDIT if needed
exportVideo = true;                                   % write a video of the animation
videoFile   = fullfile(dataDir, 'satellite_pursuit.mp4');
frameRate   = 20;
trailLen    = 80;                                     % trailing samples to draw (Inf = full)

%% ------------------------- load ---------------------------
meta  = readtable(fullfile(dataDir, 'meta.csv'));
sat   = readtable(fullfile(dataDir, 'sat_trajectory.csv'));
defs  = readtable(fullfile(dataDir, 'defenders.csv'));

flag  = [meta.flag_x, meta.flag_y];
Rkeep = meta.R_KEEPOUT;
capR  = meta.GOAL_THRESH;
nDef  = meta.num_defenders;

% --- Display convention: x-axis = in-track (positive pointing LEFT, via XDir reverse below);
% y-axis = radial. Swap the loaded columns once here so every plot call below is a plain
% (x = in-track, y = radial) with no per-call swapping. Distances are swap-invariant. ---
tmp = sat.x;  sat.x  = sat.y;  sat.y  = tmp;
tmp = defs.x; defs.x = defs.y; defs.y = tmp;
flag = flag([2 1]);

times   = unique(sat.t, 'stable');
nFrames = numel(times);
thc     = linspace(0, 2*pi, 60);

% Square, equal-aspect view. CW motion drifts far more along-track (y) than
% radial (x), so framing to the raw data with equal aspect squishes x into a thin
% strip. Instead center on the data and use one common half-span for both axes, so
% the plot stays square (keep-out circles remain circular) and fills the window.
allx = [sat.x; defs.x; flag(1)];
ally = [sat.y; defs.y; flag(2)];
pad  = 150;
cx   = 0.5 * (min(allx) + max(allx));
cy   = 0.5 * (min(ally) + max(ally));
half = 0.5 * max(max(allx) - min(allx), max(ally) - min(ally)) + pad;
xl   = [cx - half, cx + half];
yl   = [cy - half, cy + half];

%% ------------------------- figure -------------------------
fig = figure('Color', 'w', 'Position', [100 100 820 820]);
ax  = axes(fig); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
xlim(ax, xl); ylim(ax, yl);
daspect(ax, [1 1 1]);   % equal data units on both axes; square limits above keep the box square
                        % and let it scale with the window (avoid axis-equal's limit fiddling)
set(ax, 'XDir', 'reverse');   % in-track increases to the LEFT
xlabel(ax, 'In-track  [m]  (+ left)'); ylabel(ax, 'Radial  [m]');
title(ax, 'KinoPaxSTAR CostPrune \cdot satellite flag capture');

satCol = [0.10 0.40 0.90];
defCol = [0.85 0.30 0.30];

% flag + capture radius
plot(ax, flag(1), flag(2), 'p', 'MarkerSize', 20, 'MarkerFaceColor', [0.95 0.80 0.10], 'MarkerEdgeColor', 'k');
plot(ax, flag(1) + capR*cos(thc), flag(2) + capR*sin(thc), ':', 'Color', [0.5 0.5 0.5]);

% handles updated per frame
hSatTrail = plot(ax, nan, nan, '-',  'Color', satCol, 'LineWidth', 1.6);
hFlyout   = plot(ax, nan, nan, '-',  'Color', [0.55 0.70 0.95], 'LineWidth', 1.0);  % noisy open-loop fly-out of the whole plan
hPlan     = plot(ax, nan, nan, '--', 'Color', satCol, 'LineWidth', 1.4);            % nominal planned trajectory (smooth ghost)
hSat      = plot(ax, nan, nan, 'o',  'MarkerSize', 10, 'MarkerFaceColor', satCol, 'MarkerEdgeColor', 'k');
hDef      = gobjects(nDef, 1); hDefKeep = gobjects(nDef, 1); hDefTrail = gobjects(nDef, 1);
for d = 1:nDef
    hDefTrail(d) = plot(ax, nan, nan, '-', 'Color', defCol, 'LineWidth', 0.7);
    hDefKeep(d)  = plot(ax, nan, nan, '-', 'Color', defCol);
    hDef(d)      = plot(ax, nan, nan, 'o', 'MarkerSize', 6, 'MarkerFaceColor', defCol, 'MarkerEdgeColor', 'k');
end
hTxt = text(ax, 0.02, 0.97, '', 'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 11, 'BackgroundColor', [1 1 1], 'EdgeColor', [0.8 0.8 0.8]);
legend([hSatTrail, hPlan, hFlyout], {'flown', 'planned (nominal)', 'fly-out (noisy)'}, ...
       'Location', 'southoutside', 'Orientation', 'horizontal', 'AutoUpdate', 'off');

%% ------------------------- video --------------------------
useGif = false; vw = [];
if exportVideo
    try
        vw = VideoWriter(videoFile, 'MPEG-4'); vw.FrameRate = frameRate; open(vw);
    catch
        videoFile = strrep(videoFile, '.mp4', '.gif'); useGif = true;
    end
end

%% ------------------------- animate ------------------------
planCache   = containers.Map('KeyType', 'double', 'ValueType', 'any');
flyoutCache = containers.Map('KeyType', 'double', 'ValueType', 'any');
for f = 1:nFrames
    t   = times(f);
    idx = find(sat.t == t, 1);
    cyc = sat.cycle(idx);

    upto = (sat.t <= t);
    setTrail(hSatTrail, sat.x(upto), sat.y(upto), trailLen);
    set(hSat, 'XData', sat.x(idx), 'YData', sat.y(idx));

    P = getCurve(dataDir, cyc, planCache, 'plan');       % smooth nominal plan
    if isempty(P), set(hPlan, 'XData', nan, 'YData', nan);
    else,          set(hPlan, 'XData', P(:,1), 'YData', P(:,2)); end
    F = getCurve(dataDir, cyc, flyoutCache, 'flyout');   % example noisy fly-out of the whole plan
    if isempty(F), set(hFlyout, 'XData', nan, 'YData', nan);
    else,          set(hFlyout, 'XData', F(:,1), 'YData', F(:,2)); end

    dnow = defs(defs.t == t, :);
    for d = 1:nDef
        dd = dnow(dnow.id == (d-1), :);
        if isempty(dd), continue; end
        set(hDef(d),     'XData', dd.x(1),                     'YData', dd.y(1));
        set(hDefKeep(d), 'XData', dd.x(1) + Rkeep*cos(thc),    'YData', dd.y(1) + Rkeep*sin(thc));
        dtr = defs(defs.id == (d-1) & defs.t <= t, :);
        setTrail(hDefTrail(d), dtr.x, dtr.y, trailLen);
    end

    d2flag = hypot(sat.x(idx) - flag(1), sat.y(idx) - flag(2));
    set(hTxt, 'String', sprintf('t = %5.0f s    cycle %d    dist to flag = %.0f m', t, cyc, d2flag));
    if d2flag < capR, set(hTxt, 'Color', [0.1 0.6 0.1]); end
    drawnow;

    if exportVideo
        fr = getframe(fig);
        if useGif
            [A, map] = rgb2ind(frame2im(fr), 256);
            if f == 1, imwrite(A, map, videoFile, 'gif', 'LoopCount', Inf, 'DelayTime', 1/frameRate);
            else,      imwrite(A, map, videoFile, 'gif', 'WriteMode', 'append', 'DelayTime', 1/frameRate); end
        else
            writeVideo(vw, fr);
        end
    end
end
if exportVideo && ~useGif, close(vw); end
fprintf('Done (%d frames). Output: %s\n', nFrames, videoFile);

%% ------------------------- helpers ------------------------
function setTrail(h, x, y, n)
    if numel(x) > n, x = x(end-n+1:end); y = y(end-n+1:end); end
    set(h, 'XData', x, 'YData', y);
end

function P = getCurve(dataDir, cyc, cache, prefix)
    if isKey(cache, cyc), P = cache(cyc); return; end
    fn = fullfile(dataDir, 'plans', sprintf('%s_cycle%d.csv', prefix, cyc));
    P = [];
    if isfile(fn)
        T = readtable(fn);
        if height(T) > 0, P = [T.y, T.x]; end   % [in-track, radial] to match the swapped display axes
    end
    cache(cyc) = P;
end
