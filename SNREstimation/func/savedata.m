% Add noise to the entire file in sequence according to the signal-to-noise ratio estimation result
function [] = DoAddNoise(file_out,file_in,fileName,signal_power,noisevalue,target)
    
    file = fullfile(file_in,fileName);

    fileInfo = dir(file); 
    fileSize = fileInfo.bytes; 
    times = fileSize /4/ 3e6;  
    filepathOut = fullfile(file_out, fileName(1:end-4) + "-noise");
    if ~exist(filepathOut,"dir")
        mkdir(filepathOut);
    end
    filepathOut = fullfile(filepathOut, num2str(target) + "dB.dat");
    for i = 1:times
        fp = fopen(file,"rb"); 
        fseek(fp,3e6*(i-1),-1);
        data = fread(fp,3e6,"float");
        fclose(fp);
        dataIQ = data(1:2:end) + 1j*data(2:2:end);
        noisy_data = awgn1(dataIQ,noisevalue,signal_power);   
        if(i == 1)
            fp = fopen(filepathOut,"wb+");
        else
            fp = fopen(filepathOut,"ab+");
        end
        data=zeros(length(dataIQ)*2,1);
        data(1:2:end) = real(noisy_data);
        data(2:2:end) = imag(noisy_data);
        fwrite(fp,data,"float");
        fclose(fp);
    end
end

