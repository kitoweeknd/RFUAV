% all
%% Multiple frequency resolutions and color mapping plotting
clc;clear;close all;
% args
fs = 100e6;                     % sample rate  
fftpoint = [128,256,512,1024];
time_sec = 0.1;
dataform = 'float32';
byte_per = 4;
datalength = time_sec*fs*byte_per*2;
file_in ="";
filepathOut = "";

files_dat = dir(fullfile(file_in, '*.dat'));
files_iq = dir(fullfile(file_in, '*.iq'));
files = [files_dat; files_iq];

for ii = 1:length(files)
    fileName{ii} = files(ii).name;
    file_input = fullfile(file_in,fileName{ii});
%     % Determine the drone model from the file name
%     myname = char(fileName{ii});
%     for i = 1:length(myname)
%         if strcmp(myname(i), '-')
%             flytype = string(myname(1:i-1));
%             break;
%         end
%     end
    filepathOut_get = filepathOut + '\' + fileName{ii}(1:end-4);
%     filepathOut = filepathOut + flytype + '\' + fileName{ii}(1:end-3);
    color = ["parula","hsv","hot","autumn"];

    fp = fopen(file_input, 'rb');
    fseek(fp, 0, 1);
    fileSize = ftell(fp);
    fclose(fp);
    readtime = ceil(fileSize/datalength);

    time = 0;
    for i =1:readtime
        tic
        fp = fopen(file_input, 'rb');
        fseek(fp,(i-1)*datalength,-1);
        data = fread(fp,datalength/4,dataform);
        fclose(fp);
        dataIQ = data(1:2:end-1) + 1i * data(2:2:end);
        clear data;
        for j = 1:length(fftpoint)
            for k = 1:length(color)
                stft(dataIQ,fs,FFTLength=fftpoint(j));
                colormap(color(k));
                yticks([-50 :10:50]);
                yticklabels([5710:10:5810]);
                xticks([0:10:100]);
                xticklabels([0:0.1/10:0.1]);
                xlabel("Time(s)");
                title(fileName{ii}(1:end-4));
%                 title(flytype);
                filepathOut = filepathOut_get + '\'+ color(k) + '\'+num2str(fftpoint(j));
                if ~exist(filepathOut,"dir")
                    mkdir(filepathOut);
                end
                newFile = fullfile(filepathOut,num2str(2+i*0.1-0.1+time) + "-" + num2str(2+i*0.1+time) + "s-" +...
                    color(k) + "-" + num2str(fftpoint(j)) + ".jpg");
                set(gcf, 'Units', 'inches', 'Position', [0, 0, 8, 6]);
                print(gcf, newFile, '-dpng', '-r300');
                clf;
            end
        end
        toc
    end
end