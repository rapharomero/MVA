function [ yp yt ] = predict_svm_reg( Xtrain,Ytrain,Xtest,G,gamma,lambda)
%FIT_SVM_REG Fits a svm regressor to X train and Ytrain, with the regularization parameter regparam, and computes the
% predicted values on Xtest.
%   gamma is the RBF scale parameter
% 

for l=1:G
    for j = 1:G
        
        %disp((j+(l-1)*G)/(G^2)); 
        yij=Ytrain(:,l,j);
        Mdl = fitrsvm(Xtrain,yij,'Standardize',true,'KernelFunction','rbf','KernelScale',1/(2*sqrt(gamma)),'BoxConstraint',lambda);
        yp(:,l,j)=predict(Mdl,Xtest);
        yt(:,l,j)=predict(Mdl,Xtrain);
    end
end

end

