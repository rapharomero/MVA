function [  ] = plot_distr( y,q )
%PLOT_DISTR Plots the emotion distribution given in y, as well as the
%   annotations given in q
%   y must be of shape G*G, representing the emotion distribution of one
%   particular song
%   q must have two columns
% 
G=size(y,2);
disp(G);
x1=(repmat(1:G,1,G)-1/2)*2/G-1;
x2=(repmat(1:G,G,1)-1/2)*2/G-1;

scatter3(x1(:),x2(:),y(:),30,'b','filled');
hold on;
scatter(q(:,1),q(:,2),'r','filled');

legend('Kernel estimation','Annotations','Location','northeast');
xlabel('valence');
ylabel('arousal');

xlim([-1 1]);
ylim([-1 1]);
end

