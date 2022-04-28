%%% scrappylilscript just to montage a few images for myself
%%% William Plucknett, 2022
%% Scalograms
IMGDIR = "D:\thewi\Documents\UM\WN22\ML\Project\Datasets\ml-project\output\figs\magnitude scalograms\labeled axes\";
%% montage scalo c3
c3DIR = IMGDIR + "C3\";
imds3 = imageDatastore(c3DIR, 'FileExtensions', '.png');
figure(1)
montage(imds3, 'BackgroundColor', 'w')
saveas(1, IMGDIR + "C3_montage.png")
%% montage scalo c4
c4DIR = IMGDIR + "C4\";
imds4 = imageDatastore(c4DIR, 'FileExtensions', '.png');
figure(2)
montage(imds4, 'BackgroundColor', 'w')
saveas(2, IMGDIR + "C4_montage.png")


%% DFT
DFTDIR = 'D:\thewi\Documents\UM\WN22\ML\Project\Datasets\ml-project\output\figs\fta\avg only\';
imdsDFT = imageDatastore(DFTDIR, 'FileExtensions', '.png');
figure(3)
montage(imdsDFT, 'BackgroundColor', 'w')
saveas(3, DFTDIR + "DFT_montage.png")