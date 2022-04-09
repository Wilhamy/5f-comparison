% process raw kaya
% Inputs: DATADIR: directory containing the data
%       OUTDIR: output directory
%       filename: name of .mat file
% Outputs: Saves file in .mat located at OUTDIR\FTA_filename
% Note: the examples are sorted by label.
function [] = raw_process(DATADIR, OUTDIR, filename)

load(DATADIR + filename) % loads into o

d = [0;diff(o.marker)];
d(d < 0) = 0;
d(d > 5) = 0;
[cnt_uniq, uniqu_d] = hist(d, unique(d));
[uniqu_d, cnt_uniq']
locs = find(d); % indexes of the onsets

onset_win_s = [0, 0.85]; % seconds relative to onset
onset_win_n = onset_win_s * o.sampFreq; % number of datapoints relative to onset

win_len = diff(onset_win_n); % length of window, in samples
lpf_bound_f = 5; % [Hz] upper bound of the lowerpass filter
lpf_bound_i = ceil(5 / (o.sampFreq / win_len)) ; % [samples] the greatest index the lowpass filter includes (assuming n-point fourier transform where n is the win_len)

rel_channels = [1:21]; %relevant channels, removes A1, A2, X3
rel_chnames = o.chnames(rel_channels); % relevant channel names

examples = [];
labels = [];
for idx = 1:length(locs)
    loc = locs(idx);
    win = onset_win_n + loc;
    
    ex = o.data(win(1):win(2)-1,rel_channels);
    
    examples(end+1, :, :) = ex;
    labels(end+1,:) = d(loc);
end

[labels_sort, sort_idx] = sort(labels);
examples_sort = examples(sort_idx,:,:);

data.id = o.id;
data.sampFreq = o.sampFreq;
data.examples = examples_sort;
data.labels = labels_sort;
data.chames = rel_chnames;

save(OUTDIR + "RAW_" + filename, 'data')