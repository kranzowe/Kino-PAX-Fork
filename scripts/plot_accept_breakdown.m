%% Acceptance-Reason Breakdown — KinoPaxSTARCOMBO vs a KinoPaxSTARCleanCost reference
% Reads the per-iteration CSVs written by examples/gpu/kinopaxstar_accept_breakdown.cu
% (run via scripts/run_accept_breakdown.sh).
%
% WHAT THIS ANSWERS. The tuning sweeps show THAT the tuning changes outcomes, not WHY. A candidate
% enters the frontier through one of three doors, and the normal output cannot tell them apart:
%
%   min-cost   the region-best exemption   (cost <= minCostsR1[r])          both planners
%   seed       the R2 seeding exemption    CleanCost only, off by default; REMOVED in COMBO
%   roll       CleanCost: rand < cap*(w*pSyclop + (1-w)*pCost + floor)
%              COMBO:     rand < min(pMax, comboShape * pTargetAccept)
%
% The roll is a single Bernoulli draw against a SUM, so "accepted because of X" is not a distinction
% the rule makes. Each accepted node instead splits one unit of credit in proportion to each term's
% share, so the three credits sum to acc_roll by construction. The three terms differ per planner:
% CleanCost splits across (syclop, cost, floor), COMBO across its three sigmoids (cov, col, cst).
%
% AND WHY IT ALSO PLOTS THE BUDGET (figure 4). For COMBO the acceptance split is only half the rule
% -- the growth controller sets the scale and the fan-out. Figure 4 is the one to read when
% propagate falls onto kernel2 early; see its header for what each panel means.
%
% ANTI-CLUTTER RULE: COLOUR ENCODES THE ACCEPTANCE REASON and POSITION ENCODES THE POINT in figures
% 1-3. There are only five reasons, so one fixed palette is reused and the eye learns it once.
% Figure 4 is different -- it overlays points, so there colour encodes the PROFILE and line style
% the GAIN, matching process_combo_tuning_and_plot.m.
%
% AXIS NOTE: everything is plotted against ITERATION, never wall-clock. The counting atomics distort
% timing, so elapsed time from this binary is not comparable to the tuning sweep's.
%
% USAGE: cd into the data directory, then call the script BY NAME, not via run():
%   cd build/Data/Benchmarks/KinoPaxStarAcceptBreakdown/house
%   addpath('<repo>/scripts')
%   plot_accept_breakdown

clear; clc; close all;

%% --- Configuration ---
dataDir   = '';                 % '' = current directory
env       = 'house';
envTitle  = 'House';
delta     = 'large_length';     % delta + cost metric token, as in the filenames

% Must mirror PROFILES / GAINS / REACT_FRAC and the CleanCost reference in
% kinopaxstar_accept_breakdown.cu (integer label tokens: 100 x the float, as in the filenames).
comboProfiles = {'none', 'cov', 'col', 'cst', 'all'};
comboGains    = [0 100 400 1600];
comboRf       = 10;
cleanRef      = [90 100 3];     % [w k cap] tokens -- the low-cap reference run

% Five reasons, one fixed palette, reused in figures 1-3. The three credit slots mean different
% terms for the two planners, which the legend spells out.
catNames  = {'min-cost', 'seed', 'credit A (syclop|cov)', 'credit B (cost|col)', 'rejected'};
catColors = [0.85 0.33 0.10;    % min-cost   - burnt orange
             0.93 0.69 0.13;    % seed       - amber (expected 0)
             0.20 0.42 0.69;    % credit A   - steel blue
             0.30 0.64 0.36;    % credit B   - green
             0.72 0.72 0.72];   % rejected   - grey

% Figure 4 channels: colour = profile, style = gain. Matches the tuning-sweep plot.
profileColors = [0.15 0.15 0.15;    % none (control)
                 0.13 0.55 0.45;    % cov
                 0.60 0.25 0.60;    % col
                 0.85 0.45 0.10;    % cst
                 0.24 0.45 0.70];   % all
gainStyles = {':', '-', '--'};      % gain = 1, 4, 16
cleanColor = [0.70 0.15 0.20];

%% --- Build the flat point list: the reference first, then the COMBO grid ---
% P(i) = struct(label, name, isCombo, color, style)
P = struct('label', {}, 'name', {}, 'isCombo', {}, 'color', {}, 'style', {});

P(end + 1) = struct( ...
    'label',   sprintf('KinoPaxSTARCleanCost_r2off_w%d_k%d_cap%d', cleanRef(1), cleanRef(2), cleanRef(3)), ...
    'name',    sprintf('CleanCost w%g k%g cap%g', cleanRef(1)/100, cleanRef(2)/100, cleanRef(3)/100), ...
    'isCombo', false, 'color', cleanColor, 'style', '-');

for pi = 1:numel(comboProfiles)
    for gi = 1:numel(comboGains)
        prof = comboProfiles{pi};
        gval = comboGains(gi);
        % Mirror comboSkip(): 'none' owns gain 0 and nothing else.
        if xor(strcmp(prof, 'none'), gval == 0), continue; end
        if gval == 0
            sty = '-';
        else
            sty = gainStyles{find(comboGains(2:end) == gval, 1)};
        end
        P(end + 1) = struct( ...
            'label',   sprintf('KinoPaxSTARCOMBO_%s_g%d_rf%d', prof, gval, comboRf), ...
            'name',    sprintf('COMBO %s g%g', prof, gval/100), ...
            'isCombo', true, 'color', profileColors(pi, :), 'style', sty); %#ok<SAGROW>
    end
end
nP = numel(P);

%% --- Load every point ---
T = cell(1, nP);
for i = 1:nP
    fn = sprintf('%s_%s_delta%s_run0.csv', env, P(i).label, delta);
    fp = fullfile(dataDir, fn);
    if isfile(fp)
        T{i} = readtable(fp);
    else
        fprintf('  MISSING: %s\n', fn);
    end
end

% Small-multiple grid that fits the flat point list.
nCols = 5;
nRows = ceil(nP / nCols);

%% ====================== FIGURE 1: run-total composition ======================
% One normalized bar per point: what fraction of all collision-free candidates entered by each door,
% summed over the whole run. The single "how do the points differ" view.
figure('Name', sprintf('%s - Acceptance composition (run totals)', envTitle), ...
       'Position', [60 60 1500 620]);
hold on;

M      = zeros(nP, 5);
labels = cell(1, nP);
for i = 1:nP
    labels{i} = P(i).name;
    t = T{i};
    if isempty(t), continue; end
    tot = sum(t.prop_valid);
    if tot <= 0, continue; end
    [cA, cB] = creditPair(t, P(i).isCombo);
    M(i, :) = [sum(t.acc_min_cost), sum(t.acc_seed), sum(cA), sum(cB), sum(t.rejected)] / tot;
end

hb = bar(M, 'stacked', 'EdgeColor', 'none');
for c = 1:5, hb(c).FaceColor = catColors(c, :); end
set(gca, 'XTick', 1:nP, 'XTickLabel', labels, 'FontSize', 7);
xtickangle(60);
xlim([0.5, nP + 0.5]); ylim([0 1]);
ylabel('fraction of collision-free candidates'); grid on;
legend(catNames, 'Location', 'eastoutside');
xline(1.5, 'k-', 'LineWidth', 1.2, 'HandleVisibility', 'off');   % reference | COMBO grid
title(sprintf(['Acceptance composition \\x2014 %s, run totals\n' ...
               'colour = acceptance reason, position = point; bar 1 is the CleanCost reference'], envTitle), ...
      'FontWeight', 'bold');

%% ====================== FIGURE 2: composition over iterations ======================
% Same five categories, same colours, as a normalized stacked area vs iteration -- shows how the mix
% EVOLVES. The one to watch is min-cost: it dominates early and should fade as regions fill, and if
% it does not, the exemption is the planner and the sigmoids are decoration.
figure('Name', sprintf('%s - Acceptance composition vs iteration', envTitle), ...
       'Position', [70 70 1650 850]);
for i = 1:nP
    subplot(nRows, nCols, i);
    t = T{i};
    if isempty(t), axis off; continue; end
    denom = max(t.prop_valid, 1);
    [cA, cB] = creditPair(t, P(i).isCombo);
    A = [t.acc_min_cost, t.acc_seed, cA, cB, t.rejected] ./ denom;
    ha = area(t.iteration, A, 'EdgeColor', 'none');
    for c = 1:5, ha(c).FaceColor = catColors(c, :); end
    ylim([0 1]); xlim([1 max(t.iteration)]);
    set(gca, 'FontSize', 6);
    title(P(i).name, 'FontSize', 7);
    if i == 1, ylabel('fraction', 'FontSize', 7); end
    if i == nP
        legend(catNames, 'Location', 'eastoutside', 'FontSize', 6);
        xlabel('iteration', 'FontSize', 7);
    end
end
sgtitle(sprintf('Acceptance composition vs iteration \\x2014 %s', envTitle), ...
        'FontSize', 11, 'FontWeight', 'bold');

%% ====================== FIGURE 3: throughput ======================
% Kept SEPARATE from the composition figures on purpose: the scale differences between points are
% enormous, and mixing them into a normalized plot would hide both effects. Log-y.
figure('Name', sprintf('%s - Throughput vs iteration', envTitle), ...
       'Position', [80 80 1650 850]);
for i = 1:nP
    subplot(nRows, nCols, i);
    t = T{i};
    if isempty(t), axis off; continue; end
    accepted = t.acc_min_cost + t.acc_seed + t.acc_roll;
    semilogy(t.iteration, max(t.prop_attempted, 1), '-', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.1); hold on;
    semilogy(t.iteration, max(t.prop_valid, 1),     '-', 'Color', [0.20 0.42 0.69], 'LineWidth', 1.1);
    semilogy(t.iteration, max(accepted, 1),         '-', 'Color', [0.30 0.64 0.36], 'LineWidth', 1.4);
    grid on; xlim([1 max(t.iteration)]);
    set(gca, 'FontSize', 6);
    title(P(i).name, 'FontSize', 7);
    if i == nP
        legend({'propagated (attempts)', 'collision-free', 'accepted'}, ...
               'Location', 'eastoutside', 'FontSize', 6);
        xlabel('iteration', 'FontSize', 7);
    end
end
sgtitle(sprintf(['Throughput vs iteration \\x2014 %s (log y)\n' ...
                 'the gap between blue and green is what the acceptance budget throttles'], envTitle), ...
        'FontSize', 11, 'FontWeight', 'bold');

%% ====================== FIGURE 4: budget and frontier composition ======================
% READ THIS FIRST WHEN PROPAGATE FALLS ONTO KERNEL2 EARLY.
%
%   1  prop_attempted / frontier_repeat_size -- exactly 32 on the kernel1 path (one 32-thread block
%      per repeat entry), less on kernel2. h_propIterations_ alone is NOT a valid detector: it is
%      only assigned inside the kernel2 branch, so on kernel1 it holds a stale value.
%   2  reactivated / frontier_size -- Part B's share of the frontier. Part B re-activates the region
%      best UNCONDITIONALLY, one per explored region, outside h_reactFrac_ entirely, so F has a hard
%      floor at nActive. Near 100% here means F is region-best dominated.
%   3  n_active against frontier_size -- the floor itself, drawn against what the frontier actually
%      is. When the two curves meet, the region-best guarantee IS the frontier.
%   4  the two acceptance budgets and the fan-out target (COMBO only).
%
% Since rep >= 1 always, frontierRepeatSize >= F, so kernel2 is forced once 32*F > remaining no
% matter what repTarget does. If panels 2 and 3 show F pinned to nActive, no acceptance tuning will
% move the kernel1 crossover -- the levers are the region-best guarantee or the R1 grid size.
figure('Name', sprintf('%s - Budget and frontier composition', envTitle), ...
       'Position', [90 90 1650 860]);

subplot(2, 2, 1); hold on;
for i = 1:nP
    t = T{i};
    if isempty(t), continue; end
    fr = col(t, 'frontier_repeat_size');
    pa = col(t, 'prop_attempted');
    if isempty(fr) || isempty(pa), continue; end
    ok = fr > 0;
    plot(t.iteration(ok), pa(ok) ./ fr(ok), P(i).style, 'Color', P(i).color, ...
         'LineWidth', 1.2, 'DisplayName', P(i).name);
end
yline(32, 'k--', 'kernel1', 'LineWidth', 1.2, 'HandleVisibility', 'off');
ylim([0 36]); grid on;
xlabel('iteration'); ylabel('propagations per repeat entry');
title('Kernel1 check: 32 while the ceiling holds, below = kernel2');

subplot(2, 2, 2); hold on;
for i = 1:nP
    t = T{i};
    if isempty(t), continue; end
    re = col(t, 'reactivated');
    if isempty(re), continue; end
    ok = t.frontier_size > 0;
    plot(t.iteration(ok), 100 * re(ok) ./ double(t.frontier_size(ok)), P(i).style, ...
         'Color', P(i).color, 'LineWidth', 1.2, 'DisplayName', P(i).name);
end
ylim([0 105]); grid on;
xlabel('iteration'); ylabel('% of frontier from Part B');
title('Frontier composition: near 100% = region-best dominated');

subplot(2, 2, 3); hold on;
for i = 1:nP
    t = T{i};
    if isempty(t), continue; end
    na = col(t, 'n_active');
    if isempty(na), continue; end
    plot(t.iteration, t.frontier_size, P(i).style, 'Color', P(i).color, 'LineWidth', 1.2, ...
         'DisplayName', sprintf('%s: frontier', P(i).name));
    plot(t.iteration, na, ':', 'Color', min(P(i).color + 0.35, 1), 'LineWidth', 1.0, ...
         'HandleVisibility', 'off');
end
set(gca, 'YScale', 'log'); grid on;
xlabel('iteration'); ylabel('nodes / regions');
title({'frontier\_size (solid) vs n\_active (dotted)', 'curves meeting = the guarantee IS the frontier'});

subplot(2, 2, 4); hold on;
for i = 1:nP
    t = T{i};
    if ~P(i).isCombo || isempty(t), continue; end
    pt = col(t, 'p_target_accept');
    pr = col(t, 'p_target_reactivate');
    if isempty(pt), continue; end
    plot(t.iteration, pt, P(i).style, 'Color', P(i).color, 'LineWidth', 1.2, ...
         'DisplayName', sprintf('%s: p\\_accept', P(i).name));
    if ~isempty(pr)
        plot(t.iteration, pr, ':', 'Color', min(P(i).color + 0.35, 1), 'LineWidth', 1.0, ...
             'HandleVisibility', 'off');
    end
end
set(gca, 'YScale', 'log'); grid on;
xlabel('iteration'); ylabel('acceptance budget');
title({'p\_target\_accept (solid), p\_target\_reactivate (dotted)', 'flat = pinned at pMax, i.e. demand exceeds supply'});
legend('Location', 'eastoutside', 'FontSize', 6);

sgtitle(sprintf('Budget and frontier composition \\x2014 %s', envTitle), ...
        'FontSize', 11, 'FontWeight', 'bold');

%% ====================== consistency report ======================
% The identities the CSV must satisfy. The runner already checks these per iteration and exits
% non-zero on failure; re-checking here catches a stale or mismatched CSV.
fprintf('\n--- consistency ---\n');
bad = 0;
for i = 1:nP
    t = T{i};
    if isempty(t), continue; end
    part = t.acc_min_cost + t.acc_seed + t.acc_roll + t.rejected - t.prop_valid;
    if any(part ~= 0)
        fprintf('  PARTITION FAIL: %s\n', P(i).name); bad = bad + 1;
    end
    [cA, cB] = creditPair(t, P(i).isCombo);
    cC   = creditThird(t, P(i).isCombo);
    cred = cA + cB + cC - t.acc_roll;
    if any(abs(cred) > 1e-3 * max(t.acc_roll, 1))
        fprintf('  CREDIT FAIL:    %s\n', P(i).name); bad = bad + 1;
    end
    if any(t.acc_seed ~= 0)
        fprintf('  UNEXPECTED SEED ACCEPTS: %s\n', P(i).name); bad = bad + 1;
    end
end
if bad == 0, fprintf('  all checks passed.\n'); end

% The headline number this whole runner exists to produce.
fprintf('\n--- kernel1 crossover (first iteration with prop/repeat < 32) ---\n');
for i = 1:nP
    t = T{i};
    if isempty(t), continue; end
    fr = col(t, 'frontier_repeat_size'); pa = col(t, 'prop_attempted');
    if isempty(fr) || isempty(pa), continue; end
    ok  = fr > 0;
    idx = find(ok & (pa ./ max(fr, 1)) < 31.5, 1);
    if isempty(idx)
        fprintf('  %-28s never  (stayed on kernel1 for all %d iterations)\n', P(i).name, height(t));
    else
        fprintf('  %-28s iteration %d of %d   (tree_size %d)\n', ...
                P(i).name, t.iteration(idx), height(t), t.tree_size(idx));
    end
end
fprintf('\n4 figures generated.\n');

%% --- helpers ---
function v = col(t, name)
    % Column by name, or [] when the CSV predates it. Mirrors getCol in the tuning-sweep plot:
    % appended columns stay backward compatible, so an older CSV just draws fewer series.
    if ismember(name, t.Properties.VariableNames), v = double(t.(name)); else, v = []; end
end

function [a, b] = creditPair(t, isCombo)
    % The first two credit terms for whichever planner wrote this run. CleanCost splits across
    % (syclop, cost, floor); COMBO across its three sigmoids (cov, col, cst).
    if isCombo
        a = col(t, 'credit_cov'); b = col(t, 'credit_col');
    else
        a = col(t, 'credit_syclop'); b = col(t, 'credit_cost');
    end
    n = height(t);
    if isempty(a), a = zeros(n, 1); end
    if isempty(b), b = zeros(n, 1); end
end

function c = creditThird(t, isCombo)
    if isCombo, c = col(t, 'credit_cst'); else, c = col(t, 'credit_floor'); end
    if isempty(c), c = zeros(height(t), 1); end
end
