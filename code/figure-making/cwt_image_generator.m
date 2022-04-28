%% Load the data
load cwt\CWT_NoSpatial_5F-SubjectC-151204-5St-SGLHand % loads into `data`

disp('Examples size:')
disp(size(data.examples))

%% directories
OUTDIR = "D:\thewi\Documents\UM\WN22\ML\Project\Datasets\ml-project\output\figs\";
%%
uniq_labels = unique(data.labels);
avg_classes = {};
bs = [0,0];
tax = linspace(0, 0.85, 170); % 170 time points, 0.85 spanned
fax = linspace(0, 100, 46); % 46 freq points, 100 Hz spanned (200 Fs)
for i = 1:length(uniq_labels)
    label = uniq_labels(i);
    this_class = data.examples(data.labels==label,:,:,:);
    avg_class = squeeze(mean(this_class,1));
    avg_classes{end+1} = avg_class;
    %%% Specific channels %%%
    for ch_i = 1:length(data.chames)
        disp("Class " + int2str(i) + " Channel " + data.chames{ch_i})
        temp = squeeze(abs(avg_class(ch_i,:,:)));
        figure(1)
        surface(tax, fax, temp)
        colorbar
        ylabel("Frequency [Hz]")
        xlabel("Time since onset [sec]")
        xlim([0, 0.85])
%         title({"Class " + int2str(i), " Channel " + data.chames{ch_i}})
        saveas(1, OUTDIR + "Class_" + int2str(i) + " Channel_" + data.chames{ch_i} + ".png")
    end
    
    %%% Montaging %%%
%     figure(1)
%     montage(real(avg_class), 'BorderSize', bs)
%     title(['Class: ', int2str(label), ' Real'])
%     figure(2)
%     montage(imag(avg_class), 'BorderSize', bs)
%     title(['Class: ', int2str(label), ' Imaginary'])
%     figure(3)
%     montage(abs(avg_class), 'BorderSize', bs)
%     title(['Class: ', int2str(label), ' Magnitude'])
%     saveas(1, OUTDIR + "Real_" + int2str(label) + ".png")
%     saveas(2, OUTDIR + "Imag_" + int2str(label) + ".png")
%     saveas(3, OUTDIR + "Magn_" + int2str(label) + ".png")
    
end

%% messing around with pcs
X = abs(data.examples(:,:));
XXt = X * X';
[~,S,~] = svd(XXt);
s = diag(S);

mean_ev = mean(s(2:end));
count = sum(s > mean(s(2:end)))


subplot(121)
stem(s)
hold on
yline(mean_ev);
title("all evs")
subplot(122)
stem(s(2:end))
hold on
yline(mean_ev)
title("2:end evs")