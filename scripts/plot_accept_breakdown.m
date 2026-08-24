%% Acceptance-Reason Breakdown — KinoPaxSTARCleanCost
% Reads the per-iteration CSVs written by examples/gpu/kinopaxstar_accept_breakdown.cu
% (run via scripts/run_accept_breakdown.sh).
%
% WHAT THIS ANSWERS. The tuning sweeps show THAT w, k and cap change outcomes, not WHY. A candidate
% enters the frontier through one of three doors, and the normal output cannot tell them apart:
%
%   min-cost   the region-best exemption   (cost <= minCostsR1[r])
%   seed       the R2 seeding exemption    (OFF by default, so expected to be identically 0)
%   roll       the weighted roll           (rand < cap*(w*pSyclop + (1-w)*pCost + probFloor))
%
% The roll is a single Bernoulli draw against a weighted SUM, so "accepted by syclop" is not a
% distinction the rule makes. Each accepted node instead splits one unit of credit in proportion to
% each term's share -- credit_syclop + credit_cost + credit_floor == acc_roll by construction.
%
% ANTI-CLUTTER RULE: with 21 tunings, COLOUR ENCODES THE ACCEPTANCE REASON and POSITION ENCODES THE
% TUNING. There are only five reasons, so one fixed five-colour palette is reused in every figure
% and the eye learns it once; colouring by tuning instead would be unreadable.
%
% The 21 points are 7 (w,k) combinations x 3 caps, which lays out as a 3 x 7 grid of small
% multiples. 21 and not 27 because at w = 1 the cost term vanishes from weightedAccept, so only
% k = 1 is run there.
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

% Must mirror WEIGHTS / WEIGHTED_EXPS / CAPS in kinopaxstar_accept_breakdown.cu (integer label
% tokens: 100 x the float, exactly as they appear in the filenames).
weights = [90 95 100];
wExps   = [25 100 1600];
caps    = [3 10 100];

% Five reasons, one fixed palette, reused in every figure.
catNames  = {'min-cost', 'seed', 'syclop credit', 'cost credit', 'rejected'};
catColors = [0.85 0.33 0.10;    % min-cost      - burnt orange
             0.93 0.69 0.13;    % seed          - amber (expected 0)
             0.20 0.42 0.69;    % syclop credit - steel blue
             0.30 0.64 0.36;    % cost credit   - green
             0.72 0.72 0.72];   % rejected      - grey

%% --- Build the (w,k) column list, mirroring the benchmark's w=1 -> k=1 skip ---
wkList = {};   % each entry: [wToken kToken]
for wi = 1:numel(weights)
    for ei = 1:numel(wExps)
        if weights(wi) == 100 && wExps(ei) ~= 100, continue; end
        wkList{end + 1} = [weights(wi) wExps(ei)]; %#ok<SAGROW>
    end
end
nWK  = numel(wkList);
nCap = numel(caps);

%% --- Load every point: T{ci, ji} is one run's table, or [] if missing ---
T = cell(nCap, nWK);
for ci = 1:nCap
    for ji = 1:nWK
        wk    = wkList{ji};
        label = sprintf('KinoPaxSTARCleanCost_r2off_w%d_k%d_cap%d', wk(1), wk(2), caps(ci));
        fn    = sprintf('%s_%s_delta%s_run0.csv', env, label, delta);
        fp    = fullfile(dataDir, fn);
        if isfile(fp)
            T{ci, ji} = readtable(fp);
        else
            fprintf('  MISSING: %s\n', fn);
        end
    end
end

%% ====================== FIGURE 1: run-total composition ======================
% One normalized bar per tuning point. This is the single "how do the tunings differ" view:
% what fraction of all collision-free candidates entered by each door, summed over the whole run.
figure('Name', sprintf('%s - Acceptance composition (run totals)', envTitle), ...
       'Position', [60 60 1500 620]);
hold on;

M      = zeros(nCap * nWK, 5);
labels = cell(1, nCap * nWK);
row    = 0;
for ci = 1:nCap                       % cap-major so the three cap blocks group visually
    for ji = 1:nWK
        row = row + 1;
        wk  = wkList{ji};
        labels{row} = sprintf('cap%g w%g k%g', caps(ci)/100, wk(1)/100, wk(2)/100);
        t = T{ci, ji};
        if isempty(t), continue; end
        tot = sum(t.prop_valid);
        if tot <= 0, continue; end
        M(row, :) = [sum(t.acc_min_cost), sum(t.acc_seed), ...
                     sum(t.credit_syclop), sum(t.credit_cost), sum(t.rejected)] / tot;
    end
end

hb = bar(M, 'stacked', 'EdgeColor', 'none');
for c = 1:5, hb(c).FaceColor = catColors(c, :); end
set(gca, 'XTick', 1:numel(labels), 'XTickLabel', labels, 'FontSize', 7);
xtickangle(60);
xlim([0.5, numel(labels) + 0.5]); ylim([0 1]);
ylabel('fraction of collision-free candidates'); grid on;
legend(catNames, 'Location', 'eastoutside');
% Divider between cap blocks, so the grouping is unmissable.
for ci = 1:(nCap - 1)
    xline(ci * nWK + 0.5, 'k-', 'LineWidth', 1.2, 'HandleVisibility', 'off');
end
title(sprintf(['Acceptance composition by tuning \\x2014 %s, run totals\n' ...
               'colour = acceptance reason, position = tuning; blocks are cap'], envTitle), ...
      'FontWeight', 'bold');

%% ====================== FIGURE 2: composition over iterations ======================
% Same five categories, same colours, but as a normalized stacked area vs iteration -- shows how the
% mix EVOLVES (e.g. min-cost dominating early, then fading as regions fill).
figure('Name', sprintf('%s - Acceptance composition vs iteration', envTitle), ...
       'Position', [70 70 1650 850]);
for ci = 1:nCap
    for ji = 1:nWK
        subplot(nCap, nWK, (ci - 1) * nWK + ji);
        t = T{ci, ji};
        if isempty(t), axis off; continue; end
        denom = max(t.prop_valid, 1);
        A = [t.acc_min_cost, t.acc_seed, t.credit_syclop, t.credit_cost, t.rejected] ./ denom;
        ha = area(t.iteration, A, 'EdgeColor', 'none');
        for c = 1:5, ha(c).FaceColor = catColors(c, :); end
        ylim([0 1]); xlim([1 max(t.iteration)]);
        set(gca, 'FontSize', 6);
        if ci == 1
            wk = wkList{ji};
            title(sprintf('w%g k%g', wk(1)/100, wk(2)/100), 'FontSize', 7);
        end
        if ji == 1, ylabel(sprintf('cap %g', caps(ci)/100), 'FontSize', 8, 'FontWeight', 'bold'); end
        if ci == nCap && ji == 1, xlabel('iteration', 'FontSize', 7); end
        if ci == nCap && ji == nWK
            legend(catNames, 'Location', 'eastoutside', 'FontSize', 6);
        end
    end
end
sgtitle(sprintf('Acceptance composition vs iteration \\x2014 %s (rows = cap, cols = w,k)', envTitle), ...
        'FontSize', 11, 'FontWeight', 'bold');

%% ====================== FIGURE 3: throughput ======================
% Kept SEPARATE from the composition figures on purpose: the scale differences between caps are
% enormous, and mixing them into a normalized plot would hide both effects. Log-y.
figure('Name', sprintf('%s - Throughput vs iteration', envTitle), ...
       'Position', [80 80 1650 850]);
for ci = 1:nCap
    for ji = 1:nWK
        subplot(nCap, nWK, (ci - 1) * nWK + ji);
        t = T{ci, ji};
        if isempty(t), axis off; continue; end
        accepted = t.acc_min_cost + t.acc_seed + t.acc_roll;
        semilogy(t.iteration, max(t.prop_attempted, 1), '-',  'Color', [0.35 0.35 0.35], 'LineWidth', 1.1); hold on;
        semilogy(t.iteration, max(t.prop_valid, 1),     '-',  'Color', [0.20 0.42 0.69], 'LineWidth', 1.1);
        semilogy(t.iteration, max(accepted, 1),         '-',  'Color', [0.30 0.64 0.36], 'LineWidth', 1.4);
        grid on; xlim([1 max(t.iteration)]);
        set(gca, 'FontSize', 6);
        if ci == 1
            wk = wkList{ji};
            title(sprintf('w%g k%g', wk(1)/100, wk(2)/100), 'FontSize', 7);
        end
        if ji == 1, ylabel(sprintf('cap %g', caps(ci)/100), 'FontSize', 8, 'FontWeight', 'bold'); end
        if ci == nCap && ji == 1, xlabel('iteration', 'FontSize', 7); end
        if ci == nCap && ji == nWK
            legend({'propagated (attempts)', 'collision-free', 'accepted'}, ...
                   'Location', 'eastoutside', 'FontSize', 6);
        end
    end
end
sgtitle(sprintf(['Throughput vs iteration \\x2014 %s (log y; rows = cap, cols = w,k)\n' ...
                 'the gap between blue and green is what cap throttles'], envTitle), ...
        'FontSize', 11, 'FontWeight', 'bold');

%% ====================== consistency report ======================
% The identities the CSV must satisfy. The runner already checks these per iteration and exits
% non-zero on failure; re-checking here catches a stale or mismatched CSV.
fprintf('\n--- consistency ---\n');
bad = 0;
for ci = 1:nCap
    for ji = 1:nWK
        t = T{ci, ji};
        if isempty(t), continue; end
        part = t.acc_min_cost + t.acc_seed + t.acc_roll + t.rejected - t.prop_valid;
        cred = t.credit_syclop + t.credit_cost + t.credit_floor - t.acc_roll;
        if any(part ~= 0)
            fprintf('  PARTITION FAIL: cap%g w%g k%g\n', caps(ci)/100, wkList{ji}(1)/100, wkList{ji}(2)/100);
            bad = bad + 1;
        end
        if any(abs(cred) > 1e-3 * max(t.acc_roll, 1))
            fprintf('  CREDIT FAIL:    cap%g w%g k%g\n', caps(ci)/100, wkList{ji}(1)/100, wkList{ji}(2)/100);
            bad = bad + 1;
        end
        if any(t.acc_seed ~= 0)
            fprintf('  UNEXPECTED SEED ACCEPTS (r2 should be off): cap%g w%g k%g\n', ...
                    caps(ci)/100, wkList{ji}(1)/100, wkList{ji}(2)/100);
            bad = bad + 1;
        end
        % At w = 1 the cost term is multiplied by (1-w) = 0, so it can never earn credit.
        if wkList{ji}(1) == 100 && any(t.credit_cost > 1e-6)
            fprintf('  COST CREDIT AT w=1 (must be 0): cap%g k%g\n', caps(ci)/100, wkList{ji}(2)/100);
            bad = bad + 1;
        end
    end
end
if bad == 0, fprintf('  all checks passed.\n'); end
fprintf('\n3 figures generated.\n');
