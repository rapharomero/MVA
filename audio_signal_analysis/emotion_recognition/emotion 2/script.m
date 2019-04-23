load ntumir-60/C.mat ;
load ntumir-60/L.mat ;
load ntumir-60/P.mat ;
load ntumir-60/PY.mat ;
load ntumir-60/X.mat ;
load ntumir-60/Y.mat ;


%%
% Plot annotations examples
%Annotations associated with the first song set
% As1(:,2*k-1:2*k) corresponds the annotations of the k-th song in the
% first dataset
As1=PY(C==1,:); 
scatter_va(As1(:,1:16)); % Only the first 8 songs for plot clarity

%%
% Example of ground truth construction via Kernel Density Estimation
G=8;
q=As1(:,5:6);
x1=(repmat(1:G,1,G)-1/2)*2/G-1;
x2=(repmat(1:G,G,1)-1/2)*2/G-1;
y=kernel_est(q,G,0.03,0.04);

plot_distr(y,q);

%%
% Prediction (Leave-one-out strategy)

% Compute ground truth from the user annotations
G=8; % Grid size
[y]=ground_truth(PY,C,G);
n=size(y,1);
% Split dataset 
idxs=randperm(n);
train_idxs=idxs(1:n-1);
test_idxs=idxs(n);
% Kernel SVM parameters
gamma=0.001;
lambda=0.01;
% Prediction
yp_final=prediction(X,y,train_idxs,test_idxs,gamma,lambda);


% Plot the predicted distribution for the first song
idx=test_idxs(1);
q=annotations(PY,C,idx);
plot_distr(yp_final(1,:,:),q);

% Compute the final determination coefficient  
ytest=y(idx,:,:);
R2=computeR2(ytest,yp_final);
%%
% Average on all the songs in the set (Leave-one-out strategy)

%Shuffle the songs
idxs=randperm(n);


% Kernel SVM parameters
gamma=0.001;
lambda=0.01;

%For any song in the dataset
R2s=zeros(n,1)
for k = 1:n
    train_idxs=idxs([1:k k+2:n]);
    test_idxs=idxs(k);
    % Prediction
    yp=prediction(X,y,train_idxs,test_idxs,gamma,lambda);

    % Compute the final determination coefficient  
    ytest=y(idx,:,:);
    R2s(k)=computeR2(ytest,yp);
end

R2_mean=mean(R2s);