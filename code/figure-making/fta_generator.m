%% Load the data
load fta\FTA_5F-SubjectC-151204-5St-SGLHand % loads into `data`

disp('Examples size:')
disp(size(data.examples))

%% directories
OUTDIR = "D:\thewi\Documents\UM\WN22\ML\Project\Datasets\ml-project\output\figs\fta\";
%%
uniq_labels = unique(data.labels);
avg_classes = {};
% tax = linspace(0, 0.85, 170); % 170 time points, 0.85 spanned
% fax = linspace(0, 100, 46); % 46 freq points, 100 Hz spanned (200 Fs)
for i = 1:length(uniq_labels)
    label = uniq_labels(i);
    this_class = data.examples(data.labels==label,:);
    avg_class = squeeze(mean(this_class,1));
    avg_classes{end+1} = avg_class;
    
    std_class = squeeze(std(this_class, 1));
    
    %%% Plot mean only
%     disp("Class " + int2str(i));
%     figure(1)
%     plot(avg_class)
%     saveas(1, OUTDIR + "Class_" + int2str(i) + ".png")
    %%% Plot error bars
    disp("Class " + int2str(i));
    figure(1)
    plot(avg_class, 'LineWidth', 3.0, 'Color', [0 0.4470 0.7410])
    hold on
    errorbar(avg_class, std_class, 'LineWidth', 1.0, 'Color', [0 0.4470 0.7410])
    hold off
    saveas(1, OUTDIR + "Class_" + int2str(i) + ".png")
end