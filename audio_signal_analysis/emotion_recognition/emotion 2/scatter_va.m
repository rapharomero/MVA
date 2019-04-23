function [  ] = scatter_va( Ann )
%UNTITLED Summary of this function goes here
%   Plots the annotations given in Ann
% Ann must be of size 40*30
idxs=1:size(Ann,2)/2;


for idx = idxs
    c=idx*ones(size(Ann,1),1);
    scatter(Ann(:,2*idx-1),Ann(:,2*idx),50,c,'filled');
    hold on;
end

labels = repmat('Song ',8,1);
labels = labels+string(idxs');
legend(labels,'Location','northeast');

xlim([-1 1]);
xlabel('valence');
ylim([-1 1]);
ylabel('arousal');
yL = get(gca,'YLim');
xL = get(gca,'Xlim');


line([0 0],yL,'Color','b');
line(xL,[0 0],'Color','b');
end

