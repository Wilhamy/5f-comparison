%% Load the data
load cwt\CWT_NoSpatial_5F-SubjectC-151204-5St-SGLHand % loads into `data`

disp('Examples size:')
disp(size(data.examples))

%%
uniq_labels = unique(data.labels);
avg_classes = {};
for i = 1:length(uniq_labels)
    label = uniq_labels(i);
    this_class = data.examples(data.labels==label,:,:,:);
    avg_class = squeeze(mean(this_class,1));
    avg_classes{end+1} = avg_class;
    figure
    montage(avg_class)
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