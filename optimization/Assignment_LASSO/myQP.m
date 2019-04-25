function f = myQP(x,Q,p)
%computes the value of the objective function of the inequality constrained
%qradratic problem.
    f = 0.5*x'*Q*x + p'*x;
end