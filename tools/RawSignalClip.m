%% Split large data files based on the time required
clc;
clear;
close all;

% args
fs = 100e6;                     % sample rate
time_sec = 0.3;                 % seconds per output chunk
dataform = 'float32';
byte_per = 4;
datalength = time_sec*fs*byte_per*2;   % bytes per chunk (I+Q interleaved)
file_input ="";  % path
[filepath, name, ~] = fileparts(file_input);
filepath = fullfile(filepath, name);

if ~exist(filepath,"dir")
    mkdir(filepath);
else
    disp("file exist!");
end

% Open the input once; seek per chunk and close after the loop.
fp = fopen(file_input, 'rb');
fseek(fp, 0, 1);
fileSize = ftell(fp);
readtime = ceil(fileSize/datalength);

for i = 1:readtime
    tic
    fseek(fp,(i-1)*datalength,-1);
    data = fread(fp,datalength/4,dataform);
    % Label each chunk by its actual time span (derived from time_sec).
    newFile = fullfile(filepath, name + '_' + num2str((i-1)*time_sec) + '-' + num2str(i*time_sec) + 's.iq');
    fp_write = fopen(newFile,"wb+");
    fwrite(fp_write,data,dataform);
    fclose(fp_write);
    toc
end
fclose(fp);
