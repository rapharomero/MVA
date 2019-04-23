function A = autocorrelation(I,x,y)
% Inputs:
%   I : 2*2 RGB image
%   x,y : coordinates of the point where the autocorrelation is considered
    [Ix Iy] = imageDerivative(I,1);
    A = [Ix(x,y)*Ix(x,y) Iy(x,y)*Ix(x,y);
         Iy(x,y)*Ix(x,y) Iy(x,y)*Iy(x,y)]
    
end