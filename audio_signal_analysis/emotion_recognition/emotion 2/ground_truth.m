function [ y,qs ] = ground_truth( PY,C ,G)
%GROUND_TRUTH Computes the ground truth values given the annotaions in PY,
%   the song sets id given in C, and the grid size G
% The returned values are:
%           y the estimated discrete distribution over the G*G grid
%           qs as the annotations per song 
% 
%   


s_per_set=15;
%For each song set
for k=1:4
    As=PY(C==k,:);
    % For each song in the song set
    for j=1:s_per_set
        s=s_per_set*(k-1)+j;
        
        q=As(:,2*j-1:2*j);
        y(s,:,:)=kernel_est(q,G,0.03,0.04);
    end
end

end

