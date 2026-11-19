close all
clc
clear all

try
    % Define root directory
    rootDir = '/home/owkr8158/Kino-PAX-Fork';
    
    % Enable batch mode rendering
    fprintf('Starting MATLAB script in batch mode...\n');
    fprintf('Root directory: %s\n', rootDir);

    % Parameters
    numFiles = 11;  % Adjust this to match your actual number of parent files
    radius = .05;
    N = 8;
    n = 4;
    sampleSize = 17;
    stateSize = 12;
    controlSize = 3;

    xGoal = [.80, .95, .90];
    alpha = .7;
    STEP_SIZE = .2;
    model = 3;

    % Obstacle file path (relative to root)
    obstacleFilePath = fullfile(rootDir, 'include/config/obstacles/quadTrees/obstacles.csv');
    fprintf('Reading obstacles from: %s\n', obstacleFilePath);
    obstacles = gpuArray(readmatrix(obstacleFilePath));
    fprintf('Loaded %d obstacles\n', size(obstacles, 1));

    % Tree size path (relative to build/Data)
    treeSizePath = fullfile(rootDir, 'build/Data/TreeSize/TreeSize0/treeSize.csv');
    fprintf('Reading tree sizes from: %s\n', treeSizePath);
    treeSizes = gpuArray(readmatrix(treeSizePath));
    treeSizes = [0; treeSizes];

    colors = gpuArray([0 0 1;  % Blue
                    0 .9 .2;  % Green
                    1 0 1;  % Pink
                    .7 .7 0;  % Yellow
                    0 .7 .7; % Turquoise
                    1 .5 0]); % Orange

    % Create figs directory if it doesn't exist
    figsDir = fullfile(rootDir, 'figs');
    if ~exist(figsDir, 'dir')
        fprintf('Creating figs directory: %s\n', figsDir);
        mkdir(figsDir);
    else
        fprintf('Figs directory exists: %s\n', figsDir);
    end

    % Define cube vertices once
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
        1, 2;
        2, 3;
        3, 4;
        4, 1;
        5, 6;
        6, 7;
        7, 8;
        8, 5;
        1, 5;
        2, 6;
        3, 7;
        4, 8
    ];

    %% ========== PART 1: CREATE GIF WITH SPARSE PLOTTING ==========
    fprintf('\n========== CREATING TREE EXPANSION GIF ==========\n');
    
    % Create a single figure for tree expansion GIF that persists across all files
    fprintf('Creating tree expansion animation figure...\n');
    figExpansion = figure('Position', [100, 100, 1000, 1000], 'Visible', 'off');
    hold on;
    axis equal;
    axis off;
    
    % Set up the expansion figure with scene elements (only once)
    % Read first sample file to get starting point
    sampleFilePath = fullfile(rootDir, 'build/Data/Samples/Samples0/samples1.csv');
    samples = gpuArray(readmatrix(sampleFilePath));
    
    plot3(gather(samples(1,1)), gather(samples(1,2)), gather(samples(1,3)), 'ko', 'MarkerFaceColor', 'b', 'MarkerSize', 10);
    
    % Add boundary cube
    for k = 1:size(cubeEdges, 1)
        plot3([cubeVertices(cubeEdges(k, 1), 1), cubeVertices(cubeEdges(k, 2), 1)], ...
            [cubeVertices(cubeEdges(k, 1), 2), cubeVertices(cubeEdges(k, 2), 2)], ...
            [cubeVertices(cubeEdges(k, 1), 3), cubeVertices(cubeEdges(k, 2), 3)], ...
            'k-', 'LineWidth', .05);
    end
    
    % Add goal sphere
    [Xsphere, Ysphere, Zsphere] = sphere(20);
    surf(radius * Xsphere + xGoal(1), radius * Ysphere + xGoal(2), radius * Zsphere + xGoal(3), ...
        'FaceColor', 'g', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    
    % Add obstacles
    for j = 1:size(obstacles, 1)
        x_min = obstacles(j, 1);
        y_min = obstacles(j, 2);
        z_min = obstacles(j, 3);
        x_max = obstacles(j, 4);
        y_max = obstacles(j, 5);
        z_max = obstacles(j, 6);
        vertices = gpuArray([
            x_min, y_min, z_min;
            x_max, y_min, z_min;
            x_max, y_max, z_min;
            x_min, y_max, z_min;
            x_min, y_min, z_max;
            x_max, y_min, z_max;
            x_max, y_max, z_max;
            x_min, y_max, z_max]);
        faces = gpuArray([
            1, 2, 6, 5;
            2, 3, 7, 6;
            3, 4, 8, 7;
            4, 1, 5, 8;
            1, 2, 3, 4;
            5, 6, 7, 8]);
        patch('Vertices', gather(vertices), 'Faces', gather(faces), 'FaceColor', 'r', 'EdgeColor', 'k', 'FaceAlpha', alpha);
    end
    
    camlight('headlight'); 
    camlight('right');
    lighting phong;
    view(3);
    
    % GIF filename for tree expansion (single GIF for all iterations)
    gifFilename = fullfile(figsDir, 'tree_expansion_complete.gif');
    fprintf('Creating complete tree expansion GIF: %s\n', gifFilename);
    frameCount = 0;
    
    % Capture initial frame (empty tree)
    drawnow;
    frame = getframe(figExpansion);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    imwrite(imind, cm, gifFilename, 'gif', 'Loopcount', inf, 'DelayTime', 0.3);
    frameCount = frameCount + 1;
    fprintf('Added initial frame to GIF\n');
    
    % Maximum nodes to plot per file
    MAX_NODES_PER_FILE = 5000;
    
    for i = 1:numFiles
        fprintf('\n=== Processing file %d/%d for GIF ===\n', i, numFiles);
        
        % Sample and parent file paths (relative to build/Data)
        sampleFilePath = fullfile(rootDir, sprintf('build/Data/Samples/Samples0/samples%d.csv', i));
        parentFilePath = fullfile(rootDir, sprintf('build/Data/Parents/Parents0/parents%d.csv', i));

        samples = gpuArray(readmatrix(sampleFilePath));
        parentRelations = gpuArray(readmatrix(parentFilePath));
        
        % Get the tree size for THIS iteration (file i)
        if i <= length(treeSizes)
            currentTreeSize = treeSizes(i);
        else
            currentTreeSize = Inf; % If we don't have tree size data, make all nodes blue
        end
        fprintf('Tree size for iteration %d: %d\n', i, gather(currentTreeSize));
        
        % Count valid nodes (non -1 entries)
        validIndices = find(parentRelations ~= -1);
        totalNodes = length(validIndices);
        
        % Determine sampling rate
        if totalNodes > MAX_NODES_PER_FILE
            % Sample to get approximately MAX_NODES_PER_FILE nodes
            samplingRate = ceil(totalNodes / MAX_NODES_PER_FILE);
            fprintf('Total nodes: %d, sampling every %d nodes to plot ~%d\n', totalNodes, samplingRate, MAX_NODES_PER_FILE);
        else
            samplingRate = 1;
            fprintf('Total nodes: %d, plotting all\n', totalNodes);
        end
        
        % Plot tree edges with sparse sampling and straight line segments
        nodesPlotted = 0;
        nodeCounter = 0;
        
        for j = 2:size(parentRelations, 1)
            % Check if this marks the end of a run (skip -1 entries)
            if parentRelations(j) == -1
                continue;
            end
            
            nodeCounter = nodeCounter + 1;
            
            % Skip nodes based on sampling rate
            if mod(nodeCounter, samplingRate) ~= 0
                continue;
            end
            
            % Determine color based on tree size (frontier vs established tree)
            if j > currentTreeSize
                colorIndex = 3;  % Pink for frontier
            else
                colorIndex = 1;  % Blue for established tree
            end
            
            % Get parent and child positions (just x,y,z - no propagation needed)
            parentIdx = parentRelations(j) + 1;
            parentPos = samples(parentIdx, 1:3);
            childPos = samples(j, 1:3);
            
            % Plot straight line from parent to child
            plot3([gather(parentPos(1)), gather(childPos(1))], ...
                  [gather(parentPos(2)), gather(childPos(2))], ...
                  [gather(parentPos(3)), gather(childPos(3))], ...
                  'k-', 'LineWidth', 0.01);
            
            % Plot child node
            plot3(gather(childPos(1)), gather(childPos(2)), gather(childPos(3)), 'o', ...
                'Color', gather(colors(colorIndex, :)), 'MarkerFaceColor', gather(colors(colorIndex, :)), 'MarkerSize', 1);
            
            nodesPlotted = nodesPlotted + 1;
        end
        
        % Capture ONE frame per parent file after all nodes are plotted
        drawnow;
        frame = getframe(figExpansion);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);
        imwrite(imind, cm, gifFilename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.5);
        frameCount = frameCount + 1;
        fprintf('File %d: Plotted %d/%d nodes, captured frame %d\n', i, nodesPlotted, totalNodes, frameCount);
    end
    
    % Add final solution path to GIF
    fprintf('\n=== Adding solution path to GIF ===\n');
    controlPath = fullfile(rootDir, 'build/Data/ControlPathToGoal/ControlPathToGoal0/controlPathToGoal.csv');
    controls = gpuArray(flipud(readmatrix(controlPath)));
    sampleFilePath = fullfile(rootDir, 'build/Data/Samples/Samples0/samples1.csv');
    samples = gpuArray(readmatrix(sampleFilePath));
    controls = [samples(1,:); controls];
    
    for j = 2:size(controls, 1)
        x0 = controls(j-1, 1:stateSize);
        sample = controls(j,:);
        if model == 1
            [segmentX, segmentY, segmentZ] = propDoubleIntegrator(x0, sample, STEP_SIZE, stateSize, sampleSize);
        elseif model == 2
            [segmentX, segmentY, segmentZ] = propDubinsAirplane(x0, sample, STEP_SIZE, stateSize, sampleSize);
        elseif model == 3
            [segmentX, segmentY, segmentZ] = propQuad(x0, sample, STEP_SIZE, stateSize, sampleSize);
        end
        plot3(gather(segmentX), gather(segmentY), gather(segmentZ), 'Color', 'g', 'LineWidth', 2);
    end
    drawnow;
    frame = getframe(figExpansion);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    imwrite(imind, cm, gifFilename, 'gif', 'WriteMode', 'append', 'DelayTime', 1.0);
    frameCount = frameCount + 1;
    fprintf('Added final path to GIF\n');
    
    % Close the expansion figure
    close(figExpansion);
    fprintf('\n=== GIF COMPLETE ===\n');
    fprintf('Tree expansion GIF saved with %d total frames\n', frameCount);
    
    %% ========== PART 2: CREATE STATIC JPG IMAGES FOR SOLUTION PATH ==========
    fprintf('\n========== CREATING STATIC JPG IMAGES ==========\n');
    fprintf('Creating static images for solution path...\n');
    
    % Read control path
    controlPath = fullfile(rootDir, 'build/Data/ControlPathToGoal/ControlPathToGoal0/controlPathToGoal.csv');
    controls = gpuArray(flipud(readmatrix(controlPath)));
    
    % Read initial sample to get starting point
    sampleFilePath = fullfile(rootDir, 'build/Data/Samples/Samples0/samples1.csv');
    samples = gpuArray(readmatrix(sampleFilePath));
    controls = [samples(1,:); controls];
    
    % Create figure for static images
    fig = figure('Position', [100, 100, 1000, 1000], 'Visible', 'off'); 
    hold on;
    axis equal;
    axis off;
    
    % Plot starting point
    plot3(gather(samples(1,1)), gather(samples(1,2)), gather(samples(1,3)), 'ko', 'MarkerFaceColor', 'b', 'MarkerSize', 10);
    
    % Add boundary cube
    for k = 1:size(cubeEdges, 1)
        plot3([cubeVertices(cubeEdges(k, 1), 1), cubeVertices(cubeEdges(k, 2), 1)], ...
            [cubeVertices(cubeEdges(k, 1), 2), cubeVertices(cubeEdges(k, 2), 2)], ...
            [cubeVertices(cubeEdges(k, 1), 3), cubeVertices(cubeEdges(k, 2), 3)], ...
            'k-', 'LineWidth', .05);
    end
    
    % Add goal sphere
    [Xsphere, Ysphere, Zsphere] = sphere(20);
    surf(radius * Xsphere + xGoal(1), radius * Ysphere + xGoal(2), radius * Zsphere + xGoal(3), ...
        'FaceColor', 'g', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    
    % Add obstacles
    for j = 1:size(obstacles, 1)
        x_min = obstacles(j, 1);
        y_min = obstacles(j, 2);
        z_min = obstacles(j, 3);
        x_max = obstacles(j, 4);
        y_max = obstacles(j, 5);
        z_max = obstacles(j, 6);
        vertices = gpuArray([
            x_min, y_min, z_min;
            x_max, y_min, z_min;
            x_max, y_max, z_min;
            x_min, y_max, z_min;
            x_min, y_min, z_max;
            x_max, y_min, z_max;
            x_max, y_max, z_max;
            x_min, y_max, z_max]);
        faces = gpuArray([
            1, 2, 6, 5;
            2, 3, 7, 6;
            3, 4, 8, 7;
            4, 1, 5, 8;
            1, 2, 3, 4;
            5, 6, 7, 8]);
        patch('Vertices', gather(vertices), 'Faces', gather(faces), 'FaceColor', 'r', 'EdgeColor', 'k', 'FaceAlpha', alpha);
    end
    
    camlight('headlight'); 
    camlight('right');
    lighting phong;
    
    % Plot solution path with full propagation
    fprintf('Plotting solution path with full propagation...\n');
    fprintf('Controls is size: \n', size(controls, 1));
    for j = 2:size(controls, 1)
        x0 = controls(j-1, 1:stateSize);
        sample = controls(j,:);
        if model == 1
            [segmentX, segmentY, segmentZ] = propDoubleIntegrator(x0, sample, STEP_SIZE, stateSize, sampleSize);
        elseif model == 2
            [segmentX, segmentY, segmentZ] = propDubinsAirplane(x0, sample, STEP_SIZE, stateSize, sampleSize);
        elseif model == 3
            [segmentX, segmentY, segmentZ] = propQuad(x0, sample, STEP_SIZE, stateSize, sampleSize);
        end
        plot3(gather(segmentX), gather(segmentY), gather(segmentZ), 'Color', 'g', 'LineWidth', 1);
    end
    
    % Save static images - View 3D
    view(3);
    drawnow;
    filename1 = fullfile(figsDir, 'KGMT_solution_path.jpg');
    fprintf('Saving: %s\n', filename1);
    print(gcf, filename1, '-djpeg', '-r300');
    
    % Top view
    view(2);
    drawnow;
    filename2 = fullfile(figsDir, 'top_KGMT_solution_path.jpg');
    fprintf('Saving: %s\n', filename2);
    print(gcf, filename2, '-djpeg', '-r300');
    
    % X-axis view
    midY = (min(gather(samples(:,2))) + max(gather(samples(:,2)))) / 2;
    midZ = (min(gather(samples(:,3))) + max(gather(samples(:,3)))) / 2;
    campos([0, midY, max(gather(samples(:,3))) + 1]);
    camtarget([0, midY, midZ]);
    view([-.4, -.2, 0.5]);
    drawnow;
    
    filename3 = fullfile(figsDir, 'xAxis_KGMT_solution_path.jpg');
    fprintf('Saving: %s\n', filename3);
    print(gcf, filename3, '-djpeg', '-r300');
    
    close(gcf);
    fprintf('Static images complete!\n');
    
    fprintf('\n=== SCRIPT COMPLETE ===\n');
    fprintf('Check output files in: %s\n', figsDir);


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

function [segmentX, segmentY, segmentZ] = propDoubleIntegrator(x0, sample, STEP_SIZE, stateSize, sampleSize)
    segmentX = gpuArray(x0(1));
    segmentY = gpuArray(x0(2));
    segmentZ = gpuArray(x0(3));
    u = gpuArray(sample(stateSize+1:sampleSize-1));
    duration = gpuArray(sample(sampleSize));
    numDisc = gpuArray(duration / STEP_SIZE);
    x = gpuArray(x0(1));
    y = gpuArray(x0(2));
    z = gpuArray(x0(3));
    vx = gpuArray(x0(4));
    vy = gpuArray(x0(5));
    vz = gpuArray(x0(6));
    ax = u(1);
    ay = u(2);
    az = u(3);
    for k = 1:numDisc
        x = x + (vx + (vx + 2 * (vx + ax * STEP_SIZE / 2) + (vx + ax * STEP_SIZE))) * STEP_SIZE / 6;
        y = y + (vy + (vy + 2 * (vy + ay * STEP_SIZE / 2) + (vy + ay * STEP_SIZE))) * STEP_SIZE / 6;
        z = z + (vz + (vz + 2 * (vz + az * STEP_SIZE / 2) + (vz + az * STEP_SIZE))) * STEP_SIZE / 6;
        vx = vx + (ax + 2 * ax + 2 * ax + ax) * STEP_SIZE / 6;
        vy = vy + (ay + 2 * ay + 2 * ay + ay) * STEP_SIZE / 6;
        vz = vz + (az + 2 * az + 2 * az + az) * STEP_SIZE / 6;
        segmentX = [segmentX, x];
        segmentY = [segmentY, y];
        segmentZ = [segmentZ, z];
    end
    segmentX = [segmentX, gpuArray(sample(1))];
    segmentY = [segmentY, gpuArray(sample(2))];
    segmentZ = [segmentZ, gpuArray(sample(3))];
end

function [segmentX, segmentY, segmentZ] = propDubinsAirplane(x0, sample, STEP_SIZE, stateSize, sampleSize)
    segmentX = gpuArray(x0(1));
    segmentY = gpuArray(x0(2));
    segmentZ = gpuArray(x0(3));
    u = gpuArray(sample(stateSize+1:sampleSize-1));
    duration = gpuArray(sample(sampleSize));
    numDisc = gpuArray(duration / STEP_SIZE);
    x = gpuArray(x0(1));
    y = gpuArray(x0(2));
    z = gpuArray(x0(3));
    yaw = gpuArray(x0(4));
    pitch = gpuArray(x0(5));
    v = gpuArray(x0(6));
    yawRate = u(1);
    pitchRate = u(2);
    a = u(3);
    for k = 1:numDisc
        x = x + (STEP_SIZE / 6.0) * ...
            (v * cos(pitch) * cos(yaw) + ...
            2.0 * ((v + 0.5 * STEP_SIZE * a) * cos(pitch + 0.5 * STEP_SIZE * pitchRate) * cos(yaw + 0.5 * STEP_SIZE * yawRate) + ...
                    (v + 0.5 * STEP_SIZE * a) * cos(pitch + 0.5 * STEP_SIZE * pitchRate) * cos(yaw + 0.5 * STEP_SIZE * yawRate)) + ...
            (v + STEP_SIZE * a) * cos(pitch + STEP_SIZE * pitchRate) * cos(yaw + STEP_SIZE * yawRate));
        
        y = y + (STEP_SIZE / 6.0) * ...
            (v * cos(pitch) * sin(yaw) + ...
            2.0 * ((v + 0.5 * STEP_SIZE * a) * cos(pitch + 0.5 * STEP_SIZE * pitchRate) * sin(yaw + 0.5 * STEP_SIZE * yawRate) + ...
                    (v + 0.5 * STEP_SIZE * a) * cos(pitch + 0.5 * STEP_SIZE * pitchRate) * sin(yaw + 0.5 * STEP_SIZE * yawRate)) + ...
            (v + STEP_SIZE * a) * cos(pitch + STEP_SIZE * pitchRate) * sin(yaw + STEP_SIZE * yawRate));
        
        z = z + (STEP_SIZE / 6.0) * ...
            (v * sin(pitch) + ...
            2.0 * ((v + 0.5 * STEP_SIZE * a) * sin(pitch + 0.5 * STEP_SIZE * pitchRate) + ...
                    (v + 0.5 * STEP_SIZE * a) * sin(pitch + 0.5 * STEP_SIZE * pitchRate)) + ...
            (v + STEP_SIZE * a) * sin(pitch + STEP_SIZE * pitchRate));
        
        yaw = yaw + STEP_SIZE * yawRate;
        pitch = pitch + STEP_SIZE * pitchRate;
        v = v + (STEP_SIZE / 6.0) * (a + 2.0 * (a + a) + a);
        segmentX = [segmentX, x];
        segmentY = [segmentY, y];
        segmentZ = [segmentZ, z];
    end
    segmentX = [segmentX, gpuArray(sample(1))];
    segmentY = [segmentY, gpuArray(sample(2))];
    segmentZ = [segmentZ, gpuArray(sample(3))];
end

function [segmentX, segmentY, segmentZ] = propQuad(x0, sample, STEP_SIZE, stateSize, sampleSize)
    segmentX = gpuArray(x0(1));
    segmentY = gpuArray(x0(2));
    segmentZ = gpuArray(x0(3));
    u = gpuArray(sample(stateSize+1:sampleSize-1));
    duration = gpuArray(sample(sampleSize));
    numDisc = gpuArray(duration / STEP_SIZE);
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
    
    segmentX = [segmentX, gpuArray(sample(1))];
    segmentY = [segmentY, gpuArray(sample(2))];
    segmentZ = [segmentZ, gpuArray(sample(3))];
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