function [ y ] = Schroeder(x ,Fs)
%SCHROEDER Applies Schroeder's reverb to the input signal x
%   Parameters:
%       x : Input Signal
%       Fs : Sampling Rate
%   Returns:
%       y : Transformed signal

T = 1/Fs;
Trc = 1;
Tra = 0.6;
%gains and delays for the comb filter in number of sampling periods
mc = round([0.0297,0.0371,0.0414,0.0437]/T); 
gc = 10.^((-3*mc)/(Trc*Fs));

tmp1 = zeros(length(x),1);

for k = 1:length(mc)
    tmp1 = tmp1 + comb(x,gc(k),mc(k));
end
%


%gains and delays for the all pass filter in number of sampling periods
ma = round([0.09683,0.03292]/T); 
ga = 10.^((-3*mc)/(Tra*Fs));


tmp2 = allpass(tmp1,ga(1),ma(1));
y = allpass(tmp2,ga(2),ma(2));




end

