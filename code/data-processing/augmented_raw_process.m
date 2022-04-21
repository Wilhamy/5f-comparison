% process raw kaya
% Inputs: DATADIR: directory containing the data
%       OUTDIR: output directory
%       filename: name of .mat file
% Outputs: Saves file in .mat located at OUTDIR\FTA_filename
% Note: the examples are sorted by label.
function [] = raw_process(DATADIR, OUTDIR, filename)

load(DATADIR + filename) % loads into o

offset = [-0.2, -0.1, 0, 0.1, 0.2];
d = [0;diff(o.marker)];
d(d < 0) = 0;
d(d > 5) = 0;
[cnt_uniq, uniqu_d] = hist(d, unique(d));
[uniqu_d, cnt_uniq']
locs = find(d); % indexes of the onsets % make train test split here
train_idxs = randi(size(locs,1), floor(size(locs,1) * 0.8), 1); % 80 - 20 train-test split

% train_locs = locs(train_idxs);
% test_locs = locs;
% test_locs(locs == train_locs) = [];
% 
cv = cvpartition(size(locs,1), 'HoldOut', 0.2);
test_idxs = cv.test;
% separate the splits
train_locs = locs(~test_idxs);
test_locs = locs(test_idxs);

rel_channels = [1:21]; %relevant channels, removes X3
rel_chnames = o.chnames(rel_channels); % relevant channel names
train_examples = [];
train_labels = [];
test_examples = [];
test_labels = [];

for off = offset
    onset_win_s = [0, 0.85] + off;
    onset_win_n = onset_win_s * o.sampFreq; % number of datapoints relative to onset
    for idx = 1:length(train_locs)
        train_loc = train_locs(idx);
        win = onset_win_n + train_loc;
        ex = o.data(win(1):win(2)-1,rel_channels);
        train_examples(end+1, :, :) = ex;
        train_labels(end+1,:) = d(train_loc);
    end
end
onset_win_s = [0, 0.85];
onset_win_n = onset_win_s * o.sampFreq; % number of datapoints relative to onset
for idx = 1:length(test_locs)
    test_loc = test_locs(idx);
    win = onset_win_n + test_loc;
    ex = o.data(win(1):win(2)-1,rel_channels);
    test_examples(end+1, :, :) = ex;
    test_labels(end+1, :) = d(test_loc);
end
% win_len = diff(onset_win_n); % length of window, in samples


[train_labels_sort, train_sort_idx] = sort(train_labels);
train_examples_sort = train_examples(train_sort_idx,:,:);
[test_labels_sort, test_sort_idx] = sort(test_labels);
test_examples_sort = test_examples(test_sort_idx,:,:);

data.id = o.id;
data.sampFreq = o.sampFreq;
data.train_examples = train_examples_sort;
data.train_labels = train_labels_sort;
data.test_examples = test_examples_sort;
data.test_labels = test_labels_sort;
data.chames = rel_chnames;

save(OUTDIR + "AUGtrainonly_" + filename, 'data')