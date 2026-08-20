function regen_figO4
%% REGEN_FIGO4  Regenerate FigO4 (cross-experiment) with correct Lorenz-96 values.
%  The oscillator side is read from the freshly saved oscillator_results.mat;
%  the Lorenz-96 side uses the values reported in the paper (Table tab:l96),
%  replacing the stale hard-coded [18.20, 0.00128, 10.40, 0.00111].

S = load('results/oscillator_results.mat');
results = S.results;
keys_plot = {'unedited', 'pim', 'tkm', 'pim_tkm'};
colors_plot = [0.85 0.33 0.10; 0 0.45 0.74; 0.93 0.69 0.13; 0.49 0.18 0.56];

figure('Position', [100, 100, 1200, 480]);

% (a) Lorenz-96 (40D dynamics only) -- order: Unedited, PIM, TKM, PIM+TKM
subplot(1,2,1);
l96_losses = [5.27e-1, 4.66e-5, 7.91e-1, 2.76e-5];
b = bar(l96_losses); b.FaceColor = 'flat';
for i = 1:4, b.CData(i,:) = colors_plot(i,:); end
set(gca, 'XTickLabel', {'Unedited', 'PIM', 'TKM', 'PIM+TKM'}, 'YScale', 'log');
ylabel('Val Loss (log)'); title('(a) Lorenz-96 (40D dynamics only)');
for i = 1:4
    v = l96_losses(i);
    if v < 0.1
        text(i, v*1.15, sprintf('%.2e', v), 'HorizontalAlignment', 'center', 'FontSize', 8);
    else
        text(i, v*1.15, sprintf('%.2f', v), 'HorizontalAlignment', 'center', 'FontSize', 8);
    end
end

% (b) Oscillator Network (40D + 5D control)
subplot(1,2,2);
osc_losses = zeros(1,4);
for i = 1:4, osc_losses(i) = results.(keys_plot{i}).bv; end
b = bar(osc_losses); b.FaceColor = 'flat';
for i = 1:4, b.CData(i,:) = colors_plot(i,:); end
set(gca, 'XTickLabel', {'Unedited', 'PIM', 'TKM', 'PIM+TKM'}, 'YScale', 'log');
ylabel('Val Loss (log)'); title('(b) Oscillator Network (40D + 5D control)');
for i = 1:4, text(i, osc_losses(i)*1.15, sprintf('%.2e', osc_losses(i)), 'HorizontalAlignment', 'center', 'FontSize', 8); end

sgtitle('Fig O4: Cross-Experiment -- PIM Effect (Same Ring Topology)', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigO4_Lorenz96vsOscillator.png');
close;
fprintf('FigO4 regenerated.\n');
end
