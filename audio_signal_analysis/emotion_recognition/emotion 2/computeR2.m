function [ R2 ] = computeR2( ytest,ypred )
%COMPUTER2 Computes the final R2 coefficient associated with the entire
%?performed regression.
% The test and predicted values are first vectorized and the R2 statistics 
% is computed over the obtained vectors.
% 
% R2 is a 1-D vector. It contains one value for each row of ytest.
% 


N=size(ytest,1);

R2=zeros(N);

for n =1:N
    yt=ytest(n,:,:);
    yp=ypred(n,:,:);
    % Treat the inputs as vectors
    tmp1=yt(:);
    tmp2=yp(:);
    
    % Compute determination coefficients
    v1=sum((tmp1-mean(tmp1)).^2);
    v2=sum((tmp2-mean(tmp2)).^2);
    c=sum((tmp1-mean(tmp1)).*(tmp2-mean(tmp2)));
    R2(n)=c^2/(v1*v2);
end

end

