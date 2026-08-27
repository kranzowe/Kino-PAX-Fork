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
%              COMBO:     rand < min(pMax, shape_accept * pTargetAccept)
%
% COMBO RUNS TWO SHAPES. shape_accept decides WHICH nodes join the tree; shape_fanout decides WHERE
% propagation goes, and only the latter sizes rep. Cost belongs in the first and is counter-
% productive in the second -- cost is cumulative root-to-node, so "cheap" means shallow, and
% weighting fan-out by it pours propagation around the root.
%
% The roll is a single Bernoulli draw against a SUM, so "accepted because of X" is not a distinction
% the rule makes. Each accepted node instead splits one unit of credit in proportion to each term's
% share, so the credits sum to acc_roll by construction. CleanCost splits across (syclop, cost,
% floor); COMBO across the two terms of its ACCEPTANCE shape (coverage, cost), weighted the same way
% the blend weights them -- so the credit reports not just which signal liked the node but how much
% say it had at this point in the coverage->cost slide.
%
% AND WHY IT ALSO PLOTS THE BUDGET (figures 4-8). For COMBO the acceptance split is only half the
% rule -- the growth controller sets the scale and the fan-out. Those five are the ones to read when
% propagate falls onto kernel2 early, and every quantity in them is computed from columns BOTH
% planners write, so the CleanCost reference appears in all five.
%
% ANTI-CLUTTER RULE: COLOUR ENCODES THE ACCEPTANCE REASON and POSITION ENCODES THE POINT in figures
% 1-3. There are only five reasons, so one fixed palette is reused and the eye learns it once.
% Figures 4-8 are different -- they overlay points, so there colour encodes the FAN-OUT GAIN and line
% style the ACCEPTANCE GAIN, matching process_combo_tuning_and_plot.m, with the CleanCost reference in crimson
% and drawn thicker. They are five separate figures rather than one 2x2 BECAUSE A LEGEND ATTACHES
% TO THE CURRENT AXES: a single legend on a multi-panel figure describes only the last panel, so any
% series that panel omits looks absent from the entire figure.
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

% Must mirror ACC_GAINS / FAN_GAINS / REACT_FRAC and the CleanCost reference in
% kinopaxstar_accept_breakdown.cu (integer label tokens: 100 x the float, as in the filenames).
%
% COMBO runs TWO shapes: acceptance (which nodes join) and fan-out (where propagation goes).
% kFan = 0 is the uniform-fan-out control arm -- both fan-out sigmoids pinned at 0.5, so every node
% gets identical rep, which is CleanCost's and KinoPaxPlus's behaviour.
comboAccGains = [100 400 1600];
comboFanGains = [0 100 400 1600];
comboRf       = 10;
cleanRef      = [90 100 3];     % [w k cap] tokens -- the low-cap reference run

% Five reasons, one fixed palette, reused in figures 1-3. The three credit slots mean different
% terms for the two planners, which the legend spells out.
% For COMBO the two live credit terms are the ACCEPTANCE shape's coverage and cost halves; the
% collision slot is structurally 0 now, so slot B carries cost for both planners.
catNames  = {'min-cost', 'seed', 'credit A (syclop|coverage)', 'credit B (cost|cost)', 'rejected'};
catColors = [0.85 0.33 0.10;    % min-cost   - burnt orange
             0.93 0.69 0.13;    % seed       - amber (expected 0)
             0.20 0.42 0.69;    % credit A   - steel blue
             0.30 0.64 0.36;    % credit B   - green
             0.72 0.72 0.72];   % rejected   - grey

% Figures 4-8 channels: colour = FAN-OUT gain (the headline axis), style = ACCEPTANCE gain.
% Matches process_combo_tuning_and_plot.m so the two plots read the same way.
fanColors = [0.15 0.15 0.15;    % kFan 0    (control: uniform rep)
             0.62 0.76 0.90;    % kFan 1
             0.24 0.45 0.70;    % kFan 4
             0.03 0.15 0.31];   % kFan 16   (near-bimodal, KPAX-like sparsity)
accStyles  = {':', '-', '--'};  % kAcc = 1, 4, 16
cleanColor = [0.70 0.15 0.20];

%% --- Build the flat point list: the reference first, then the COMBO grid ---
% P(i) = struct(label, name, isCombo, color, style, width)
P = struct('label', {}, 'name', {}, 'isCombo', {}, 'color', {}, 'style', {}, 'width', {});

% The reference is drawn thicker so it reads as the anchor the grid is measured against.
P(end + 1) = struct( ...
    'label',   sprintf('KinoPaxSTARCleanCost_r2off_w%d_k%d_cap%d', cleanRef(1), cleanRef(2), cleanRef(3)), ...
    'name',    sprintf('CleanCost w%g k%g cap%g', cleanRef(1)/100, cleanRef(2)/100, cleanRef(3)/100), ...
    'isCombo', false, 'color', cleanColor, 'style', '-', 'width', 2.2);

for ai = 1:numel(comboAccGains)
    for fi = 1:numel(comboFanGains)
        kAcc = comboAccGains(ai);
        kFan = comboFanGains(fi);
        if kFan == 0
            fanTag = 'uniform';
            w = 1.8;              % the control arm, drawn thicker
        else
            fanTag = sprintf('kFan%g', kFan / 100);
            w = 1.2;
        end
        P(end + 1) = struct( ...
            'label',   sprintf('KinoPaxSTARCOMBO_ka%d_kf%d_rf%d', kAcc, kFan, comboRf), ...
            'name',    sprintf('COMBO kAcc%g %s', kAcc / 100, fanTag), ...
            'isCombo', true, 'color', fanColors(fi, :), 'style', accStyles{ai}, ...
            'width',   w); %#ok<SAGROW>
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

%% ============ FIGURES 4-8: budget and frontier composition ============
% THESE ARE THE ONES TO READ WHEN PROPAGATE FALLS ONTO KERNEL2 EARLY.
%
% Deliberately five SEPARATE figures rather than one 2x2. A legend attaches to the CURRENT AXES, so
% a single legend on a multi-panel figure describes only the last panel -- and if that panel happens
% to exclude a series, the series looks absent from the whole figure. One figure per quantity means
% one clickable legend per quantity, each covering every series drawn in it.
%
% Every quantity below is computed from columns BOTH planners write, so the CleanCost reference
% appears in all five. That is the point: COMBO's controller is only meaningful against what the
% hand-tuned cap actually did.
%
% The chain these five trace out:
%   kernel1 fails  <=  32*frontierRepeatSize > remaining
%                  <=  frontierRepeatSize >= F        (rep >= 1 is a correctness clamp)
%                  <=  F >= nActive                   (region-best reactivation is UNCONDITIONAL)
% so if figure 6 shows n_active/frontier_size near 1, no acceptance tuning will move the crossover
% in figure 4 -- the levers are the region-best guarantee itself or the R1 grid size.

%% ---- FIGURE 4: kernel1 check ----
% prop_attempted / frontier_repeat_size is exactly 32 on the kernel1 path (one 32-thread block per
% repeat entry) and h_propIterations_ < 32 on kernel2. h_propIterations_ ALONE is not a valid
% detector: it is only assigned inside the kernel2 branch, so on kernel1 it holds a stale value
% from whichever earlier iteration last took kernel2.
figure('Name', sprintf('%s - Kernel1 check', envTitle), 'Position', [90 90 1150 620]);
hold on;
for i = 1:nP
    t = T{i};
    if isempty(t), continue; end
    fr = col(t, 'frontier_repeat_size');
    pa = col(t, 'prop_attempted');
    if isempty(fr) || isempty(pa), continue; end
    ok = fr > 0;
    plot(t.iteration(ok), pa(ok) ./ fr(ok), P(i).style, 'Color', P(i).color, ...
         'LineWidth', P(i).width, 'DisplayName', P(i).name);
end
yline(32, 'k--', 'kernel1', 'LineWidth', 1.2, 'HandleVisibility', 'off');
ylim([0 36]); grid on;
xlabel('iteration'); ylabel('propagations per repeat entry');
title(sprintf(['Kernel1 check \\x2014 %s\n' ...
               '32 while the ceiling holds; below 32 = kernel2 (block split across candidates)'], envTitle), ...
      'FontWeight', 'bold');
clickableLegend();

%% ---- FIGURE 5: frontier composition ----
% `reactivated` counts frontier bits among the PRE-EXISTING tree, i.e. exactly Part B's output, so
% this is the share of the frontier that is re-expansion rather than newly admitted nodes.
% CleanCost's realised value is ~75% and nobody chose it -- it fell out of the Syclop score floor.
% COMBO's h_reactFrac_ is the first time this has been an explicit knob, so this figure is the
% direct check on whether it is being honoured.
figure('Name', sprintf('%s - Frontier composition', envTitle), 'Position', [100 100 1150 620]);
hold on;
for i = 1:nP
    t = T{i};
    if isempty(t), continue; end
    re = col(t, 'reactivated');
    if isempty(re), continue; end
    ok = t.frontier_size > 0;
    plot(t.iteration(ok), 100 * re(ok) ./ double(t.frontier_size(ok)), P(i).style, ...
         'Color', P(i).color, 'LineWidth', P(i).width, 'DisplayName', P(i).name);
end
ylim([0 105]); grid on;
xlabel('iteration'); ylabel('% of frontier from Part B');
title(sprintf(['Frontier composition: reactivated / frontier\\_size \\x2014 %s\n' ...
               'near 100%% = the frontier is re-expansion, not new nodes'], envTitle), ...
      'FontWeight', 'bold');
clickableLegend();

%% ---- FIGURE 6: region-best share ----
% n_active / frontier_size. Part B re-activates the region best UNCONDITIONALLY -- one node per
% explored region, outside h_reactFrac_ entirely -- so the frontier has a hard floor at nActive.
% A curve approaching 1 means the guarantee IS the frontier, and since rep >= 1 forces
% frontierRepeatSize >= F, kernel2 then becomes unavoidable at 32*F > remaining no matter what the
% acceptance budget or repTarget do. THIS IS THE FIGURE THAT EXPLAINS FIGURE 4.
figure('Name', sprintf('%s - Region-best share of the frontier', envTitle), ...
       'Position', [110 110 1150 620]);
hold on;
for i = 1:nP
    t = T{i};
    if isempty(t), continue; end
    na = col(t, 'n_active');
    if isempty(na), continue; end
    ok = t.frontier_size > 0;
    plot(t.iteration(ok), na(ok) ./ double(t.frontier_size(ok)), P(i).style, ...
         'Color', P(i).color, 'LineWidth', P(i).width, 'DisplayName', P(i).name);
end
yline(1, 'k--', 'frontier == nActive', 'LineWidth', 1.2, 'HandleVisibility', 'off');
grid on;
xlabel('iteration'); ylabel('n\_active / frontier\_size');
title(sprintf(['Region-best share \\x2014 %s\n' ...
               'approaching 1 = the unconditional guarantee is the whole frontier'], envTitle), ...
      'FontWeight', 'bold');
clickableLegend();

%% ---- FIGURE 7: realised roll acceptance rate ----
% acc_roll / (candidates that actually took the roll). Computed the same way for both planners, so
% it is the apples-to-apples comparison between COMBO's derived pTargetAccept and CleanCost's
% hand-swept cap: it is the fraction of non-exempt candidates that were admitted.
%
% Exemptions are excluded from the denominator because they bypass the roll entirely -- including
% them would make a planner look more permissive purely for finding more region minima.
figure('Name', sprintf('%s - Realised roll acceptance rate', envTitle), ...
       'Position', [120 120 1150 620]);
hold on;
for i = 1:nP
    t = T{i};
    if isempty(t), continue; end
    rolled = double(t.prop_valid) - double(t.acc_min_cost) - double(t.acc_seed);
    ok = rolled > 0;
    if ~any(ok), continue; end
    semilogy(t.iteration(ok), max(double(t.acc_roll(ok)) ./ rolled(ok), 1e-8), P(i).style, ...
             'Color', P(i).color, 'LineWidth', P(i).width, 'DisplayName', P(i).name);
end
set(gca, 'YScale', 'log'); grid on;
xlabel('iteration'); ylabel('acc\_roll / non-exempt candidates');
title(sprintf(['Realised roll acceptance rate \\x2014 %s\n' ...
               'COMBO: should RISE over a run as the candidate pool shrinks. Flat = pinned at pMax'], envTitle), ...
      'FontWeight', 'bold');
clickableLegend();

%% ---- FIGURE 8: mean fan-out ----
% frontier_repeat_size / frontier_size -- the mean repeat count per frontier node, i.e. how many
% 32-thread blocks the average node gets. CleanCost's is the realised mean of its binary 15/1 rule;
% COMBO's is repTarget * meanShape. Both are computable from the same two columns, so this is the
% other half of the budget story and it is directly comparable.
%
% Multiply this by 32 * frontier_size to recover prop_attempted, which is the quantity the kernel1
% ceiling actually constrains.
figure('Name', sprintf('%s - Mean fan-out per frontier node', envTitle), ...
       'Position', [130 130 1150 620]);
hold on;
for i = 1:nP
    t = T{i};
    if isempty(t), continue; end
    fr = col(t, 'frontier_repeat_size');
    if isempty(fr), continue; end
    ok = t.frontier_size > 0;
    plot(t.iteration(ok), fr(ok) ./ double(t.frontier_size(ok)), P(i).style, ...
         'Color', P(i).color, 'LineWidth', P(i).width, 'DisplayName', P(i).name);
end
yline(1, 'k--', 'rep = 1 (the correctness floor)', 'LineWidth', 1.2, 'HandleVisibility', 'off');
grid on;
xlabel('iteration'); ylabel('mean repeat count per frontier node');
title(sprintf(['Mean fan-out \\x2014 %s\n' ...
               'at the floor of 1, repTarget has no room left and kernel2 is next'], envTitle), ...
      'FontWeight', 'bold');
clickableLegend();

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
fprintf('\n8 figures generated.\n');
fprintf(['  1-3  acceptance composition and throughput (colour = reason, position = point)\n' ...
         '  4-8  budget and frontier composition (colour = profile, style = gain;\n' ...
         '       CleanCost reference in crimson, drawn thicker; one clickable legend each)\n']);

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
        % COMBO's two live acceptance terms. credit_col is structurally 0 (the collision term is
        % gone), so the second slot carries COST for both planners -- which makes the stacked bars
        % directly comparable rather than accidentally plotting a dead series for one of them.
        a = col(t, 'credit_cov'); b = col(t, 'credit_cst');
    else
        a = col(t, 'credit_syclop'); b = col(t, 'credit_cost');
    end
    n = height(t);
    if isempty(a), a = zeros(n, 1); end
    if isempty(b), b = zeros(n, 1); end
end

function c = creditThird(t, isCombo)
    % The third slot exists only so the consistency check still sums to acc_roll. For COMBO that is
    % the (structurally zero) collision credit; for CleanCost it is the probability floor.
    if isCombo, c = col(t, 'credit_col'); else, c = col(t, 'credit_floor'); end
    if isempty(c), c = zeros(height(t), 1); end
end

function clickableLegend()
    % Legend whose entries toggle their series on click -- the only way 14 overlaid series stay
    % readable. Attaches to the CURRENT axes, which is why figures 4-8 are one plot each: a legend
    % on a multi-panel figure describes only the last panel, and a series missing from that panel
    % then looks absent from the whole figure. ItemHitFcn needs R2016a+; this repo is on R2023a.
    lgd = legend('Location', 'eastoutside', 'FontSize', 7);
    lgd.ItemHitFcn = @toggleSeries;
end

function toggleSeries(~, evt)
    h = evt.Peer;
    if strcmp(h.Visible, 'on')
        h.Visible = 'off';
    else
        h.Visible = 'on';
    end
end
