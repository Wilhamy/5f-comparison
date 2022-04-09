% automate process
clear;
ROOT = "D:\thewi\Documents\UM\WN22\ML\Project\Datasets\ml-project\";
DATADIR = ROOT + "original\";
OUTDIR = ROOT + "raw\";
files = dir(DATADIR + "\*.mat");
extractor = @raw_process ; % feature extractor

for idx = 1:size(files,1)
    filename = files(idx).name;
    fprintf("Processing " + filename)
    extractor(DATADIR, OUTDIR, filename) % process fta features
end