function [ R2 ] = R2Weight( ytrue,ypred )
%R2WEIGHT This function computes the weight of the regressor 
% in the fusion.
%
%   ytrue contains the ground truth ditribution
%  ypred contains the distribution predicted by the regressor being
%  weighted

G=size(ytrue,2);
R2=zeros(G);

for l = 1:G
    for j = 1:G
        tmp1=ytrue(:,l,j);
        tmp2=ypred(:,l,j);
        v1=sum((tmp1-mean(tmp1)).^2);
        v2=sum((tmp2-mean(tmp2)).^2);
        c=sum((tmp1-mean(tmp1)).*(tmp2-mean(tmp2)));

        r=c^2/(v1*v2);
        R2(l,j)=r;
    end
end

end

