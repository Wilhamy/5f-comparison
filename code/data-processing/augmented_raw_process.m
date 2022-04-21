% process raw kaya
% Inputs: DATADIR: directory containing the data
%       OUTDIR: output directory
%       filename: name of .mat file
% Outputs: Saves file in .mat located at OUTDIR\FTA_filename
% Note: the examples are sorted by label.
function [] = raw_process(DATADIR, OUTDIR, filename)

load(DATADIR + filename) % loads into o

offset = [-0.1, 0, 0.1, 0.5];
d = [0;diff(o.marker)];
d(d < 0) = 0;
d(d > 5) = 0;
[cnt_uniq, uniqu_d] = hist(d, unique(d));
[uniqu_d, cnt_uniq']
locs = find(d); % indexes of the onsets

rel_channels = [1:21]; %relevant channels, removes A1, A2, X3
rel_chnames = o.chnames(rel_channels); % relevant channel names
examples = [];
labels = [];

for off = offset
    onset_win_s = [0, 0.85] + offset;
    onset_win_n = onset_win_s * o.sampFreq; % number of datapoints relative to onset
    for idx = 1:length(locs)
        loc = locs(idx);
        win = onset_win_n + loc;
        ex = o.data(win(1):win(2)-1,rel_channels);
        examples(end+1, :, :) = ex;
        labels(end+1,:) = d(loc);
    end
end
% win_len = diff(onset_win_n); % length of window, in samples


[labels_sort, sort_idx] = sort(labels);
examples_sort = examples(sort_idx,:,:);

data.id = o.id;
data.sampFreq = o.sampFreq;
data.examples = examples_sort;
data.labels = labels_sort;
data.chames = rel_chnames;

save(OUTDIR + "AUG_" + filename, 'data')