clc;
clear;
close all;

%% ============ args ============
fs = 100e6;           % sample rate
fc = 5775e6;         % frequency center
time_sec = 0.05;     % time duration
dataformat = 'float32';

fft_range = 1:20;   % STFTP range
fftlength = 2.^fft_range;

% file path
inputFile = "file path";
outputDir = "output path";

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% ============ data input ============
blocksize = 1e6;
start_pos = 100e6;
datalength = time_sec * fs * 2;
overlap_ratio = 0.25;

fp = fopen(inputFile, 'rb');
fseek(fp, start_pos, -1);
data = fread(fp, datalength, dataformat);
fclose(fp);

dataIQ = data(1:2:end) + 1j * data(2:2:end);
clear data;

%% ============ STFT plot ============
nFFT = min(fftlength(end), numel(dataIQ));
window = hamming(nFFT);
noverlap = round(nFFT * overlap_ratio);

% frequency axis
freq_min = -50;
freq_max = 50;
freq_ticks = freq_min:10:freq_max;
freq_labels = (fc/1e6) + freq_ticks;

% time axis 
time_ticks = 0 : time_sec*1e3/10 : time_sec*1e3;
nPlots = length(fftlength);

for idx = 1:nPlots
    nfft_current = fftlength(idx);
    if nfft_current > numel(dataIQ)
        nfft_current = numel(dataIQ);
    end

    [S, F, T] = stft(dataIQ, fs, ...
        'FFTLength', nfft_current, ...
        'Window', hamming(nfft_current), ...
        'OverlapLength', round(nfft_current * overlap_ratio));

    % plot
    figure('Visible', 'off');
    surf(T*1e3, F/1e6, 20*log10(abs(S)), ...
        'EdgeColor', 'none', 'FaceAlpha', 0.9);
    view(-45, 60);
    axis tight;

    colormap autumn;
    colorbar;
    grid off;
    caxis auto;

    ytickformat('%.0f');
    xticks(time_ticks);
    yticks(freq_ticks);
    yticklabels(freq_labels);
    xlabel('Time (ms)');
    ylabel('Frequency (MHz)');
    title(sprintf('STFT (NFFT=%d)', nfft_current));

    % save plot
    filename = fullfile(outputDir, sprintf('%d-STFT.png', nfft_current));
    exportgraphics(gcf, filename, 'Resolution', 300);
    close(gcf);

    fprintf('Saved: %s\n', filename);
end

fprintf('Done! Generated %d images.\n', nPlots);
