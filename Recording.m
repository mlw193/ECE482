recObj = audiorecorder;  %initialize recording object
disp('Start recording.')
recordblocking(recObj, 30); %start of recording
disp('End of Recording.');
y = getaudiodata(recObj); %get data from audio
filename = 'recording.wav';
audiowrite(filename,y,44100); %write a 44.1 MHz sampled file of the audio
[y,Fs] = audioread('recording.wav');
window=hamming(512); %%window with size of 512 points
noverlap=256; %%the number of points for repeating the window
nfft=1024; %%size of the fADS = audioDatastore(folder)it
[S,F,T,P] = spectrogram(y,window,noverlap,nfft,Fs,'yaxis');
surf(T,F,10*log10(P),'edgecolor','none'); axis tight;view(0,90);
colormap(hot); %%for the indexed colors, check this in help for blck/white
set(gca,'clim',[-80 -30]); %%clim is the limits of the axis colours
xlabel('Time s');
ylabel('Frequency kHz')