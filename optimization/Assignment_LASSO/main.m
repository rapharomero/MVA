%Load the data
data = readtable('iris.data.txt');
data = data(strcmp(data.Var5, 'Iris-versicolor')|strcmp(data.Var5, 'Iris-virginica'),:);
%Extract data arrays from table
X = data{:,1:4};
n_obs = size(X,1);
%Add column of ones to account the offset
X = [X ones(n_obs,1)];
d = size(X,2);
%Transforms string labels into 1 or -1
y = 2*strcmp(data.Var5, 'Iris-virginica')-1;

%Disorder data to have balanced classes on training
indices = randperm(n_obs);
X = X(indices,:);
y = y(indices,:);

tau = 0.1;
tol = 10^(-5);
mu = 2;

%%
%3/3. Tests on the iris dataset

%Extract training and testing sets
Xtrain = X(1:80,:);
ytrain = y(1:80,:);
Xtest = X(81:100,:);
ytest = y(81:100,:);
%Try different value of tau
taus = logspace(-3,3,10);
scores = [];
for tau = taus
    s = score(tau,Xtrain,ytrain,Xtest,ytest,mu,tol);
    scores = [scores s];
end

semilogx(taus,scores);
title('Score for different values of tau');
xlabel('tau');
ylabel('score');
%%
%3/4. Plot semilog scaled duality gap 
%Using dampedNewton method  
tau = 0.1;
tol = 10^(-7);
mu = 2;
mus = [2,15,50,100];
for m = mus
dg = dualitygaps(tau,X,y,m,tol);
semilogy(dg);

hold on;
end
hold off;
legend('mu = 2','mu = 15','mu = 50','mu = 100');


%%
%Using backtracking linesearch method  
tau = 0.1;
tol = 10^(-7);
mu = 2;
mus = [2,15,50,100];
for m = mus
dg = dualitygaps_LS(tau,X,y,m,tol);
semilogy(dg);
hold on;
end
hold off;
legend('mu = 2','mu = 15','mu = 50','mu = 100');

