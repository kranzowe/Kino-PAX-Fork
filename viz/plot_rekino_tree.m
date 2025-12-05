close all
clc
clear all

try
    % Define root directory
    rootDir = '/home/owkr8158/Kino-PAX-Fork';
    
    fprintf('Starting ReKino tree visualization...\n');
    fprintf('Root directory: %s\n', rootDir);

    % Parameters
    radius = .05;
    alpha = .7;
    SAMPLE_DIM = 17;  % Each state is 17 elements (12 state + 4 control + 1 duration)
    STATE_DIM = 12;   % First 12 elements are state
    CONTROL_DIM = 4;  % Next 4 elements are control
    MAX_BRANCH_LENGTH = 1000;  % As defined in ReKinoLite (MAX_PATH_LENGTH)
    NUM_BRANCHES_TO_PLOT = 50;  % Plot first 50 thread branches
    STEP_SIZE = .1;   % Step size for propagation (matches STEP_SIZE in config.h)
    model = 3;        % Quadrotor model

    xGoal = [.80, .95, .90];

    % Read the ReKino tree data
    treePath = fullfile(rootDir, 'build/Data/ReKinoLiteTree/rekino_lite_tree.csv');
    fprintf('Reading tree from: %s\n', treePath);
    treeData = readmatrix(treePath);
    fprintf('Loaded tree with %d branches (rows)\n', size(treeData, 1));

    % Read branch depths to know how many nodes each branch has
    depthsPath = fullfile(rootDir, 'build/Data/ReKinoLiteTree/rekino_lite_depths.csv');
    fprintf('Reading branch depths from: %s\n', depthsPath);
    branchDepths = readmatrix(depthsPath);

    % Obstacle file path
    obstacleFilePath = fullfile(rootDir, 'include/config/obstacles/quadTrees/obstacles.csv');
    fprintf('Reading obstacles from: %s\n', obstacleFilePath);
    obstacles = readmatrix(obstacleFilePath);
    fprintf('Loaded %d obstacles\n', size(obstacles, 1));

    % Create figs directory if it doesn't exist
    figsDir = fullfile(rootDir, 'figs');
    if ~exist(figsDir, 'dir')
        fprintf('Creating figs directory: %s\n', figsDir);
        mkdir(figsDir);
    end

    % Define cube vertices for workspace boundary
    cubeVertices = [
        0, 0, 0;
        1, 0, 0;
        1, 1, 0;
        0, 1, 0;
        0, 0, 1;
        1, 0, 1;
        1, 1, 1;
        0, 1, 1
    ];
    
    cubeEdges = [
        1, 2; 2, 3; 3, 4; 4, 1;
        5, 6; 6, 7; 7, 8; 8, 5;
        1, 5; 2, 6; 3, 7; 4, 8
    ];

    % Create figure
    fig = figure('Position', [100, 100, 1000, 1000], 'Visible', 'off');
    hold on;
    axis equal;
    axis off;

    % Plot workspace boundary cube
    for k = 1:size(cubeEdges, 1)
        plot3([cubeVertices(cubeEdges(k, 1), 1), cubeVertices(cubeEdges(k, 2), 1)], ...
              [cubeVertices(cubeEdges(k, 1), 2), cubeVertices(cubeEdges(k, 2), 2)], ...
              [cubeVertices(cubeEdges(k, 1), 3), cubeVertices(cubeEdges(k, 2), 3)], ...
              'k-', 'LineWidth', 0.5);
    end

    % Plot goal sphere
    [Xsphere, Ysphere, Zsphere] = sphere(20);
    surf(radius * Xsphere + xGoal(1), radius * Ysphere + xGoal(2), radius * Zsphere + xGoal(3), ...
         'FaceColor', 'g', 'FaceAlpha', 0.5, 'EdgeColor', 'none');

    % Plot obstacles
    for j = 1:size(obstacles, 1)
        x_min = obstacles(j, 1);
        y_min = obstacles(j, 2);
        z_min = obstacles(j, 3);
        x_max = obstacles(j, 4);
        y_max = obstacles(j, 5);
        z_max = obstacles(j, 6);
        vertices = [
            x_min, y_min, z_min;
            x_max, y_min, z_min;
            x_max, y_max, z_min;
            x_min, y_max, z_min;
            x_min, y_min, z_max;
            x_max, y_min, z_max;
            x_max, y_max, z_max;
            x_min, y_max, z_max];
        faces = [
            1, 2, 6, 5;
            2, 3, 7, 6;
            3, 4, 8, 7;
            4, 1, 5, 8;
            1, 2, 3, 4;
            5, 6, 7, 8];
        patch('Vertices', vertices, 'Faces', faces, 'FaceColor', 'r', 'EdgeColor', 'k', 'FaceAlpha', alpha);
    end

    % Plot branches
    numBranchesToPlot = min(NUM_BRANCHES_TO_PLOT, size(treeData, 1));
    fprintf('\nPlotting %d branches...\n', numBranchesToPlot);

    % Generate colors for different branches
    colors = jet(numBranchesToPlot);

    for branchIdx = 1:numBranchesToPlot
        % Get this branch's data (one row)
        branchRow = treeData(branchIdx, :);
        
        % Get depth for this branch
        depth = branchDepths(branchIdx);

        % If depth is 0 or invalid, compute it from the data
        if depth == 0 || isnan(depth)
            % Find the last non-zero node
            depth = 0;
            for nodeIdx = 1:MAX_BRANCH_LENGTH
                startIdx = (nodeIdx - 1) * SAMPLE_DIM + 1;
                endIdx = startIdx + SAMPLE_DIM - 1;
                
                % Check if this node has any non-zero values
                nodeData = branchRow(startIdx:endIdx);
                if any(nodeData ~= 0)
                    depth = nodeIdx;
                else
                    % Once we hit all zeros, stop
                    break;
                end
            end
            fprintf('Branch %d: Computed depth = %d\n', branchIdx, depth);
        end
        
        % Skip branches with depth <= 1 (nothing to plot)
        if depth <= 1
            continue;
        end

        % Plot propagated edges between nodes
        for nodeIdx = 1:(depth-1)
            % Current node position in the row
            startIdx = (nodeIdx - 1) * SAMPLE_DIM + 1;
            endIdx = startIdx + SAMPLE_DIM - 1;
            
            % Next node position
            nextStartIdx = nodeIdx * SAMPLE_DIM + 1;
            nextEndIdx = nextStartIdx + SAMPLE_DIM - 1;
            
            % Extract current and next node
            currentNode = branchRow(startIdx:endIdx);
            nextNode = branchRow(nextStartIdx:nextEndIdx);
            
            % Extract state (first 12 elements)
            x0 = currentNode(1:STATE_DIM);
            
            % The sample includes the next state, control, and duration
            sample = nextNode;  % Full SAMPLE_DIM vector
            
            % Propagate using the quadrotor model
            if model == 3
                [segmentX, segmentY, segmentZ] = propQuad(x0, sample, STEP_SIZE, STATE_DIM, SAMPLE_DIM);
                
                % Plot the propagated path
                plot3(segmentX, segmentY, segmentZ, ...
                      'Color', colors(branchIdx, :), 'LineWidth', 1);
                
                % Plot node points
                plot3(x0(1), x0(2), x0(3), 'o', ...
                      'Color', colors(branchIdx, :), ...
                      'MarkerFaceColor', colors(branchIdx, :), ...
                      'MarkerSize', 2);
            end
        end
        
        % Plot final node in branch
        finalStartIdx = (depth - 1) * SAMPLE_DIM + 1;
        finalEndIdx = finalStartIdx + SAMPLE_DIM - 1;
        finalNode = branchRow(finalStartIdx:finalEndIdx);
        plot3(finalNode(1), finalNode(2), finalNode(3), 'o', ...
              'Color', colors(branchIdx, :), ...
              'MarkerFaceColor', colors(branchIdx, :), ...
              'MarkerSize', 2);
        
        if mod(branchIdx, 10) == 0
            fprintf('Plotted %d/%d branches\n', branchIdx, numBranchesToPlot);
        end
    end

    camlight('headlight'); 
    camlight('right');
    lighting phong;

    % Save 3D view
    view(3);
    drawnow;
    filename1 = fullfile(figsDir, 'rekino_tree_3d.jpg');
    fprintf('\nSaving: %s\n', filename1);
    print(gcf, filename1, '-djpeg', '-r300');

    % Save top view
    view(2);
    drawnow;
    filename2 = fullfile(figsDir, 'rekino_tree_top.jpg');
    fprintf('Saving: %s\n', filename2);
    print(gcf, filename2, '-djpeg', '-r300');

    % Save side view
    view([0, 0, 1]);
    drawnow;
    filename3 = fullfile(figsDir, 'rekino_tree_side.jpg');
    fprintf('Saving: %s\n', filename3);
    print(gcf, filename3, '-djpeg', '-r300');

    close(gcf);
    fprintf('\n=== REKINO VISUALIZATION COMPLETE ===\n');
    fprintf('Images saved to: %s\n', figsDir);

catch ME
    fprintf('ERROR OCCURRED!\n');
    fprintf('Error: %s\n', ME.message);
    fprintf('Error identifier: %s\n', ME.identifier);
    fprintf('Stack trace:\n');
    for k = 1:length(ME.stack)
        fprintf('  File: %s\n', ME.stack(k).file);
        fprintf('  Function: %s, Line: %d\n', ME.stack(k).name, ME.stack(k).line);
    end
    exit(1);
end

% Propagation function for quadrotor
function [segmentX, segmentY, segmentZ] = propQuad(x0, sample, STEP_SIZE, stateSize, sampleSize)
    segmentX = x0(1);
    segmentY = x0(2);
    segmentZ = x0(3);
    u = sample(stateSize+1:sampleSize-1);  % Extract control
    duration = sample(sampleSize);          % Extract duration
    numDisc = duration / STEP_SIZE;
    Zc = u(1);
    Lc = u(2);
    Mc = u(3);
    Nc = u(4);
    h0 = x0(1:12);
    x = x0(1);
    y = x0(2);
    z = x0(3);

    for k = 1:numDisc
        h1 = ode(h0, Zc, Lc, Mc, Nc);
        h2 = ode(h0 + 0.5 * STEP_SIZE * h1, Zc, Lc, Mc, Nc);
        h3 = ode(h0 + 0.5 * STEP_SIZE * h2, Zc, Lc, Mc, Nc);
        h4 = ode(h0 + STEP_SIZE * h3, Zc, Lc, Mc, Nc);
        
        h0 = h0 + (STEP_SIZE / 6) * (h1 + 2 * h2 + 2 * h3 + h4);
        
        x = h0(1);
        y = h0(2);
        z = h0(3);
        
        segmentX = [segmentX, x];
        segmentY = [segmentY, y];
        segmentZ = [segmentZ, z];
    end
    
    segmentX = [segmentX, sample(1)];
    segmentY = [segmentY, sample(2)];
    segmentZ = [segmentZ, sample(3)];
end

function x0dot = ode(x0, Zc, Lc, Mc, Nc)
    NU = 10e-3;
    MU = 2e-6;
    IX = 1.0;
    IY = 1.0;
    IZ = 2.0;
    GRAVITY = -9.81;
    MASS = 1.0;
    MASS_INV = 1.0 / MASS;

    phi   = x0(4);
    theta = x0(5);
    psi   = x0(6);
    u     = x0(7);
    v     = x0(8);
    w     = x0(9);
    p     = x0(10);
    q     = x0(11);
    r     = x0(12);

    x0dot = zeros(1, 12);

    x0dot(1) = cos(theta) * cos(psi) * u + (sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi)) * v + ...
            (cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)) * w;

    x0dot(2) = cos(theta) * sin(psi) * u + (sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi)) * v + ...
            (cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi)) * w;

    x0dot(3) = -sin(theta) * u + sin(phi) * cos(theta) * v + cos(phi) * cos(theta) * w;

    x0dot(4) = p + (q * sin(phi) + r * cos(phi)) * tan(theta);

    x0dot(5) = q * cos(phi) - r * sin(phi);

    x0dot(6) = (q * sin(phi) + r * cos(phi)) / cos(theta);

    XYZ = -NU * sqrt(u^2 + v^2 + w^2);
    X   = XYZ * u;
    x0dot(7)  = (r * v - q * w) - GRAVITY * sin(theta) + MASS_INV * X;

    Y  = XYZ * v;
    x0dot(8) = (p * w - r * u) + GRAVITY * cos(theta) * sin(phi) + MASS_INV * Y;

    Z  = XYZ * w;
    x0dot(9) = (q * u - p * v) + GRAVITY * cos(theta) * cos(phi) + MASS_INV * Z + MASS_INV * Zc;

    LMN = -MU * sqrt(p^2 + q^2 + r^2);
    L   = LMN * p;
    x0dot(10) = (IY - IZ) / IX * q * r + (1 / IX) * L + (1 / IX) * Lc;

    M   = LMN * q;
    x0dot(11) = (IZ - IX) / IY * p * r + (1 / IY) * M + (1 / IY) * Mc;

    N   = LMN * r;
    x0dot(12) = (IX - IY) / IZ * p * q + (1 / IZ) * N + (1 / IZ) * Nc;
end