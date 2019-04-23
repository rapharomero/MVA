[x Fs] = audioread('Sons/guitare.wav');
%%
%Phasing
%1
amin = -1;
amax = 1;
fa = 10;
B = (amax+amin)/2;
A = (amax-amin)/2;

p = 10;%Delay
 
t = (p:length(x))';
a = B+A*sin(2*pi*fa*t/Fs);% gain

x_late = x(p:end,1);

y = x(p:end) + a.*x_late;
soundsc(y,Fs)
%2
%%
%Flanger
%
n=length(x);
pmin = 0;
pmax = round(Fs/400);

a = 1;
B = (pmin+pmax)/2;
A = (pmax-pmin)/2;
t = (pmax:length(x))';
fp = 1; % frequency of oscillation of the delay
p = floor(B + A*sin(2*pi*t*fp/Fs));
delayed = t-p;
y = x(t) + a*x(delayed);
soundsc(y,Fs)
Nfft = 2^nextpow2(length(y));
Y =fft(y,Nfft);

%plot((1:Nfft)*Fs/Nfft,(abs(Y)));

%%
% 3 Artificial reverberation
%%


% First echo
S = [50 100 1000];
M = [10 0 0];
D = [10 1 100];


x1 = firstechoes(x,D,S,M,340);
y = Schroeder(x1,Fs);
soundsc(x1,Fs);
%%
