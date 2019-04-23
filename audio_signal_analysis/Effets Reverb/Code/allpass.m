function [ y ] = allpass( x,g,m)
%ALLPASS Applies all pass filter to the input signal, with gain g and delay
%m
%   Detailed explanation goes here

% Filter coefficients
a = zeros(m,1);
a(1)=1;
a(m) = -g;

b = zeros(m,1);
b(1)=-g;
b(m)=1;

y = filter(b,a,x);

end

