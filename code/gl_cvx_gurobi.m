function [x2, iter2, out2] = gl_cvx_gurobi(x0, A, b, mu, opts2)
[n,l] = size(x0);

cvx_begin quiet
    cvx_solver gurobi
    variable x(n,l)
    minimize ( 0.5*square_pos(norm(A*x-b,'fro')) + mu*sum(norms(x,2,2)) )
cvx_end

x2 = x;
iter2 = -1;
out2 = cvx_optval;
end