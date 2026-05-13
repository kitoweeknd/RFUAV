%% Split large data files based on the time required
clc;
clear;
close all;

% args
fs = 100e6;                     % sample rate
fftpoint = 8192;
time_sec = 0.3;
dataform = 'float32';
byte_per = 4;
datalength = time_sec*fs*byte_per*2;
file_input ="";  % path
[filepath, name, ~] = fileparts(file_input);
filepath = filepath + '\' + name;

if ~exist(filepath,"dir")
    mkdir(filepath);
else
    disp("file exist!");
end

fp = fopen(file_input, 'rb');
fseek(fp, 0, 1);
fileSize = ftell(fp);
fclose(fp);
readtime = ceil(fileSize/datalength);

for i =1:readtime
    tic
    fp = fopen(file_input, 'rb'); 
    fseek(fp,(i-1)*datalength,-1);
    data = fread(fp,datalength/4,dataform);
    fclose(fp);
    newFile = fullfile(filepath, name + '_' + num2str(i*2-2) + '-' + num2str(i*2) + 's.iq');
    fp_write = fopen(newFile,"wb+");
    fwrite(fp_write,data,dataform);
    fclose(fp_write);
    toc
end