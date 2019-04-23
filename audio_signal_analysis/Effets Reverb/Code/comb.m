function [ y ] = comb( x,g,m )
%COMB Applies a comb filter to the input signal x, with gain parameter g
%and delay parameter m
%   Detailed explanation goes here

%Filter coefficients
a = zeros(m,1);
a(1)=1;
a(m)=g;


b = zeros(m,1);
b(m)=1;

y = filter(b,a,x);

end

