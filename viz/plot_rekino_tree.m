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
    SAMPLE_DIM = 17;  % Each state is 17 elements
    MAX_BRANCH_LENGTH = 500;  % As defined in ReKino constructor
    NUM_BRANCHES_TO_PLOT = 50;  % Plot first 50 thread branches

    xGoal = [.80, .95, .90];

    % Read the ReKino tree data
    treePath = fullfile(rootDir, 'build/Data/ReKinoTree/rekino_tree.csv');
    fprintf('Reading tree from: %s\n', treePath);
    treeData = readmatrix(treePath);
    fprintf('Loaded tree with %d branches (rows)\n', size(treeData, 1));

    % Read branch depths to know how many nodes each branch has
    depthsPath = fullfile(rootDir, 'build/Data/ReKinoTree/rekino_depths.csv');
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

        % if we didnt set it correctly
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

        % Extract nodes from the branch (each node is SAMPLE_DIM elements)
        % Branch layout: [node_0][node_1]...[node_depth]
        for nodeIdx = 1:(depth-1)
            % Current node position in the row
            startIdx = (nodeIdx - 1) * SAMPLE_DIM + 1;
            endIdx = startIdx + SAMPLE_DIM - 1;
            
            % Next node position
            nextStartIdx = nodeIdx * SAMPLE_DIM + 1;
            nextEndIdx = nextStartIdx + SAMPLE_DIM - 1;
            
            % Extract x, y, z for current and next node (first 3 elements of each state)
            currentNode = branchRow(startIdx:endIdx);
            nextNode = branchRow(nextStartIdx:nextEndIdx);
            
            x1 = currentNode(1);
            y1 = currentNode(2);
            z1 = currentNode(3);
            
            x2 = nextNode(1);
            y2 = nextNode(2);
            z2 = nextNode(3);
            
            % Plot edge between nodes
            plot3([x1, x2], [y1, y2], [z1, z2], ...
                  'Color', colors(branchIdx, :), 'LineWidth', 1);
            
            % Plot node point
            plot3(x1, y1, z1, 'o', ...
                  'Color', colors(branchIdx, :), ...
                  'MarkerFaceColor', colors(branchIdx, :), ...
                  'MarkerSize', 2);
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