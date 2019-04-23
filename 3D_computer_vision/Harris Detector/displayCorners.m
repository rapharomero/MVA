function f = displayCorners(I,corners)
    figure;
    imshow(I);
    hold on;
    scatter(corners(:,1),corners(:,2),'bo');
    hold off;
end