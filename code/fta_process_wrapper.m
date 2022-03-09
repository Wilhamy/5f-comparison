% automate process
clear;
ROOT = "D:\thewi\Documents\UM\WN22\ML\Project\Datasets\";
DATADIR = ROOT + "raw\";
OUTDIR = ROOT + "fta\";
files = dir(DATADIR + "\*.mat");

for idx = 1:size(files,1)
    filename = files(idx).name;
    fprintf("Processing " + filename)
    fta_process(DATADIR, OUTDIR, filename) % process fta features
end