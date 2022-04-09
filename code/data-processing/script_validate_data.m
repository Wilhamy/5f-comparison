load ('../raw/RAW_5F-SubjectC-151204-5St-SGLHand.mat')

X = data.examples;
y = data.labels;

thumb = X(y==1,:,:);
pinky = X(y==5,:,:);

thumb_avg = squeeze(mean(thumb, 1));
pinky_avg = squeeze(mean(pinky, 1));

figure(1)
plot(pinky_avg(:,6))
hold on
plot(thumb_avg(:,6))
title("C3")
figure(2)
plot(pinky_avg(:,7))
hold on
plot(thumb_avg(:,7))
title("C4")
hold off