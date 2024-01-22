clear;
clc;
%Include graphing to see the plots of different methods: original 19200fft, direct 192 pts
%fft, down-sampling or pooling to make 19200 to be 192, 
M = 8;        % Modulation order (8 PSK)
fSpan = 4;     % Filter span in symbols
nData = 4096; % Number of bits 24
Fc = 1e6;     % Carrier frequency, Hz
% Assumed parameters
Fb = 1e6;      % Bit (baud) rate, bps
Fs = 3*Fc;     % Sampling frequency, Hz
% Ts = 1/Fs;     % Sample time, sec
% Td = nData/Fb; % Time duration, sec
spb = Fs/Fb;   % Samples per bit 
fft_n = 100;

for i = 1:10
    data = randi([0 M-1],nData,1);% Generate random data bits
    if i>0
        
        modData = pskmod(data,M);
        txFilter = comm.RaisedCosineTransmitFilter("RolloffFactor",0.2,"FilterSpanInSymbols",fSpan,"OutputSamplesPerSymbol",spb); %give modulated baseband signal
        txfilterOut = txFilter(modData); %"RolloffFactor",0.5,
        figure(1)
        spectrum=fftshift(abs(fft(txfilterOut, 192*fft_n)).^2);
        
        plot(spectrum);
        title('Direct fine-grained PSD');
        figure(2)
        plot(mag2db(spectrum));
        title('Direct fine-grained PSD dB');
        spectrum_dws = downsample(spectrum, fft_n);     
        figure(3)
        spectrum = nanmean(reshape([spectrum(:); nan(mod(-numel(spectrum),fft_n),1)],fft_n,[]));
        spectrum = spectrum';
        plot(mag2db(spectrum));
        title('Pooled 192 pts PSD dB');
        curtfile=sprintf('8PSK%d.mat',i);
%         save(curtfile, 'spectrum')
        figure(4)
        plot(spectrum);
        title('Pooled 192 pts PSD');       
        figure(5)
        plot( mag2db(spectrum_dws) );
        title('Downsampled 192 pts PSD dB');
        figure(6)
        spectrum_direct=fftshift(abs(fft(txfilterOut, 192)).^2);
        plot(mag2db(spectrum_direct));
        title('Direct 192 pts PSD dB')
        figure(7)
        plot(spectrum_direct);
        title('Direct 192 pts PSD')        
    end
end




% mod_raw_data=pskmod(data,M);

% figure(3)
% plot(txfilterOut)
% zplane(mod_raw_data)
