M = 4;        % Modulation order (BPSK)
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
        spectrum = nanmean(reshape([spectrum(:); nan(mod(-numel(spectrum),fft_n),1)],fft_n,[]));
        spectrum = spectrum';
        figure(2)
        plot(mag2db(spectrum));
        curtfile=sprintf('QPSK%d.mat',i);
        save(curtfile, 'spectrum')
    end
end




% mod_raw_data=pskmod(data,M);

% figure(3)
% plot(txfilterOut)
% zplane(mod_raw_data)
