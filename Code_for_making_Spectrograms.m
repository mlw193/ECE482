%%%% directions for use: pull all of the .wav file (99 of them) of desired genre into current folder and run code to get a spectrogram of Genre.XXXXX.wax('jazz.00067.wav')



%%%assuming the file is in the current folder
[y,Fs] = audioread('jazz.00067.wav');
window=hamming(512); %%window with size of 512 points
noverlap=256; %%the number of points for repeating the window
nfft=1024; %%size of the fit
[S,F,T,P] = spectrogram(y,window,noverlap,nfft,Fs,'yaxis');
surf(T,F,10*log10(P),'edgecolor','none'); axis tight;view(0,90);
colormap(hot); %%for the indexed colors, check this in help for blck/white
set(gca,'clim',[-80 -30]); %%clim is the limits of the axis colours
xlabel('Time s');
ylabel('Frequency kHz')
