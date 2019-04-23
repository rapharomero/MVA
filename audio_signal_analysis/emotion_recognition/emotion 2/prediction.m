function [ yp_final ] = prediction( X,y,train_idxs,test_idxs ,gamma,lambda)
%PREDICTION Uses the R2-weights to compute the final regressor of emotion
%   distributions
%   X contains the features ,
%   y contain the ground truth computed from the annotations
%   train_idx and test_idx contains the indexes of features to include in
%   training and testing set respectively
%   gamma and lambda are respectively the RBF scale parameter and the SVM
%   regularization parameter

%%
feature_types={'harmonic','pitch','spectral','rhythmic','temporal'};
T=length(feature_types);
n=size(y,1);
G=size(y,2);
ntest=length(test_idxs);
weights=ones(T,G,G)/T;
yp_final=ones([ntest G G]);

yps=zeros([T length(test_idxs) G G]);

for t=1:T
    type=char(feature_types(t));
    Xs=X.(type);
    Xtrain=Xs(train_idxs,:);
    Ytrain=y(train_idxs,:,:);
    Xtest=Xs(test_idxs,:);
    ytest=y(test_idxs,:,:);
    [yp, yt]=predict_svm_reg(Xtrain,Ytrain,Xtest,G,gamma,lambda); % yt is used to compute the R2 weight
    yps(t,:,:,:)=yp;
    
    % yt:output predicted on training set
    % YTrain: Training output
    R2=R2Weight(yt,Ytrain); 
                                
    weights(t,:,:)=R2;
end


for l=1:G
    for j=1:G
        weights(:,l,j)=weights(:,l,j)/sum(weights(:,l,j));
    end
end

for l=1:G
    for j=1:G
        yp_final(:,l,j)=(weights(:,l,j)'*yps(:,:,l,j));
    end
end

end

