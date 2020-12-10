function [x1, iter1, out1] = gl_cvx_mosek(x0, A, b, mu, opts1)
[n,l] = size(x0);

cvx_begin quiet
    cvx_solver mosek
    variable x(n,l)
    minimize ( 0.5*square_pos(norm(A*x-b,'fro')) + mu*sum(norms(x,2,2)) )
cvx_end

x1 = x;
iter1 = -1;
out1 = cvx_optval;
end