function dg = dualitygaps(tau,X,y,mu,tol)
n = size(X,1);
d = size(X,2);

%Optimization of the primal variable
z0 = 2*ones(n,1);
w0 = zeros(d,1);
x0 = [w0;
      z0];
[Q_primal, p_primal, A_primal, b_primal] = transform_svm_primal(tau,X,y);

[primal_sol,primal_hist] = barr_method(Q_primal, p_primal, A_primal, b_primal,x0,mu,tol);

primalvalues = arrayfun(@(k) myQP(primal_hist(:,k),Q_primal,p_primal),1:size(primal_hist,2));
%Optimization of the dual variable

lambda0 = 1/(2*tau*n) *ones(n,1);

[Q_dual, p_dual, A_dual, b_dual] = transform_svm_dual(tau,X,y);

[dual_sol,dual_hist] = barr_method(Q_dual, p_dual, A_dual, b_dual,lambda0,mu,tol);

dualvalues = -arrayfun(@(k) myQP(dual_hist(:,k),Q_dual,p_dual),1:size(dual_hist,2));

%compute the dual gap
nmin = min(size(dualvalues,2),size(primalvalues,2));
duval = dualvalues(1:nmin);
prival = primalvalues(1:nmin);
pv = prival;
dv = duval;
dg = prival - duval;
end