% process raw kaya into continuous wavelet tranformed
% Inputs: DATADIR: directory containing the data
%       OUTDIR: output directory
%       filename: name of .mat file
% Outputs: Saves file in .mat located at OUTDIR\FTA_filename
% Note: the examples are sorted by label.
function [] = cwt_process(DATADIR, OUTDIR, filename)

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

rel_channels = [1:21]; %relevant channels, removes X3
rel_chnames = o.chnames(rel_channels); % relevant channel names

examples = []; % [N, C, T, F] - number of examples; channels; time; freq TODO: verify this is the correct order of axes
labels = []; % [N] - number of examples
for idx = 1:length(locs)
    loc = locs(idx);
    win = onset_win_n + loc;
    
    ex = o.data(win(1):win(2)-1,rel_channels);
    
    % TODO: Apply CAR spatial filtering as in Khademi? %%
    
    % APPLY BANDPASS FILTER HERE %
    % bandpass 4-40 Hz for beta and mu bands per Song
    Wst = [4,40] * 2 / o.sampFreq; % normalized stopband frequencies
    [b,a] = cheby2(6, 60, Wst); % per Song et. al : 6th order, 60dB stopband attentuation
    ex = filtfilt(b, a, ex); % Will this act on the correct dimension?
    
    transformed = [];
    for ch = 1:length(rel_channels)
        wt = cwt(ex(:,ch), 'amor'); % use morlet wavelet transform
                                % TODO: include sampling frequency fs? [200]
                                % TODO: or include sampling period ts? [0.85]
        transformed(end+1, :, :) = wt;
    end
    % save the data %
    examples(end+1, :, :, :) = transformed;
    labels(end+1,:) = d(loc);
end

[labels_sort, sort_idx] = sort(labels);
examples_sort = examples(sort_idx,:,:);

data.id = o.id;
data.sampFreq = o.sampFreq;
data.examples = examples_sort;
data.labels = labels_sort;
data.chames = rel_chnames;

save(OUTDIR + "CWT_" + filename, 'data', '-v7.3') % specify v7.3 because files can be very big