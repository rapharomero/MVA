function corners_harris = harrisANMS(I,alpha,sigma,treshold,n)
% Inputs:
%   I : 2*2 RGB Image
%   alpha : sensitivity parameter
%   treshold : treshold 
%   n : number of corners to return
% Returns:
%   corners : set of harris corners on I filterned with adaptative non-max
%   suppression, in pixel coordinates
% 
%Compute the 'naive' harris corners with 
[corners_naive,values] = harrisCorners(I,alpha,sigma,treshold,1000);

% Do the ANMS 
% Initialization
% Remove first point of the remaining points and put it in the processed
% points
remaining_points = corners_naive(2:end,:);
processed_points = corners_naive(1,:);
processed_values = values(1);
% Initialize set of radiuses with infinity
radiuses = Inf;
c = 0.8;
% ANMS loops
while ~isempty(remaining_points)
    p = remaining_points(1,:);
    fp = processed_values(1,:);
    %Select the processed points q such that f(p) is below c*f(q)
    cond = fp < c*processed_values;
    feasible_points = processed_points(cond);
    % If no points have a response , the radius associated with p is +infinity
    if isempty(feasible_points)
        radiuses = [radiuses; Inf];
    % else the radius associated with p is the distance to the set of
    % feasible points.
    else
        dmin = min(sum((feasible_points-p).^2,2));
        radiuses = [radiuses ; dmin];
    end
    % Update remaining and processed points
    remaining_points = remaining_points(2:end,:);
    processed_points = [processed_points; p];
    processed_values = [processed_values; fp];
end
% Sort the points by decreasing radius
[~,rad_order] = sort(radiuses,'descend');
corners_harris = processed_points(rad_order,:);
% Truncate the result to have at most n corners
corners_harris = corners_harris(1:min(n,end),:);

end


