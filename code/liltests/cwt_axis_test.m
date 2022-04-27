close all;
clear; clc;
fs = 1/0.01;
tax = 0:(1/fs):6;

y1 = sin(1 * 2 * pi * tax) + sin(2 * 2 * pi * tax) + sin(3 * 2 * pi * tax);
y2 = cos(2 * 2 * pi * tax) + cos(4 * 2 * pi * tax) + cos(5 * 2 * pi * tax);

y = [y1; y2]';

Wst = [0.5,2] * 2 / fs; % normalized stopband frequencies
[b,a] = cheby2(6, 60, Wst); % per Song et. al : 6th order, 60dB stopband attentuation
y_tilde = filtfilt(b, a, y); % Will this act on the correct dimension?


subplot(1,2,1)
plot(y(:,1))
hold on
plot(y_tilde(:,1))
subplot(1,2,2)
plot(y(:,2))
hold on
plot(y_tilde(:,2))
