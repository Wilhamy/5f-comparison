% automate process
clear;
ROOT = "D:\thewi\Documents\UM\WN22\ML\Project\Datasets\";
DATADIR = ROOT + "original\";
OUTDIR = ROOT + "raw\";
files = dir(DATADIR + "\*.mat");
extractor = 

for idx = 1:size(files,1)
    filename = files(idx).name;
    fprintf("Processing " + filename)
    fta_process(DATADIR, OUTDIR, filename) % process fta features
end