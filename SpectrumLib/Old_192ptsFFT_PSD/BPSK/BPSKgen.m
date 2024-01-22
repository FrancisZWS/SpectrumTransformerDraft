M = 2;        % Modulation order (BPSK)
fSpan = 4;     % Filter span in symbols
nData = 4096; % Number of bits 24
Fc = 1e6;     % Carrier frequency, Hz
% Assumed parameters
Fb = 1e6;      % Bit (baud) rate, bps
Fs = 3*Fc;     % Sampling frequency, Hz
% Ts = 1/Fs;     % Sample time, sec
% Td = nData/Fb; % Time duration, sec
spb = Fs/Fb;   % Samples per bit 



for i = 1:10
    data = randi([0 M-1],nData,1);% Generate random data bits
    if i>0
        
        modData = pskmod(data,M);
        txFilter = comm.RaisedCosineTransmitFilter("RolloffFactor",0.2,"FilterSpanInSymbols",fSpan,"OutputSamplesPerSymbol",spb); %give modulated baseband signal
        txfilterOut = txFilter(modData); %"RolloffFactor",0.5,
        figure(1)
        spectrum=fftshift(abs(fft(txfilterOut, 192)).^2);
        plot(spectrum);
        figure(2)
        plot(mag2db(spectrum));
        curtfile=sprintf('BPSK%d.mat',i);
        save(curtfile, 'spectrum')
    end
end




% mod_raw_data=pskmod(data,M);

% figure(3)
% plot(txfilterOut)
% zplane(mod_raw_data)
