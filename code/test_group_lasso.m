% function test_group_lasso

% min 0.5 * ||A * x - b||_2^2 + mu * ||x||_{1,2}

% generate data
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);
n = 512;
m = 256;
A = randn(m,n);
k = round(n*0.1); l = 2;
A = randn(m,n);
p = randperm(n); p = p(1:k);
u = zeros(n,l);  u(p,:) = randn(k,l);  
b = A*u;
mu = 1e-2;
x0 = randn(n, l);

errfun = @(x1, x2) norm(x1 - x2, 'fro') / (1 + norm(x1,'fro'));
errfun_exact = @(x) norm(x - u, 'fro') / (1 + norm(u,'fro'));
sparisity = @(x) sum(abs(x(:)) > 1E-6 * max(abs(x(:)))) /(n*l);

% cvx calling mosek
opts1 = []; 
tic;
[x1, iter1, out1] = gl_cvx_mosek(x0, A, b, mu, opts1);
t1 = toc;

% cvx calling gurobi
opts2 = []; 
tic;
[x2, iter2, out2] = gl_cvx_gurobi(x0, A, b, mu, opts2);
t2 = toc;

% Subgradient Method
opts5 = []; 
tic;
[x5, iter5, out5] = gl_SGD_primal(x0, A, b, mu, opts5);
t5 = toc;

% Proximal Gradient Method for the Primal Problem
opts8 = []; 
tic;
[x8, iter8, out8] = gl_ProxGD_primal(x0, A, b, mu, opts8);
t8 = toc;

% Fast Proximal Gradient Method(Nesterov) for the Primal Problem
opts9 = []; 
tic;
[x9, iter9, out9] = gl_FProxGD_primal(x0, A, b, mu, opts9);
t9 = toc;

fprintf('     CVX-Mosek: cpu: %5.3f, iter: %5d, optval: %6.7E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t1, iter1, out1, sparisity(x1), errfun_exact(x1), errfun(x1, x1), errfun(x2, x1));
fprintf('    CVX-Gurobi: cpu: %5.3f, iter: %5d, optval: %6.7E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t2, iter2, out2, sparisity(x2), errfun_exact(x2), errfun(x1, x2), errfun(x2, x2));
fprintf('    SGD Primal: cpu: %5.3f, iter: %5d, optval: %6.7E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t5, iter5, out5.fval, sparisity(x5), errfun_exact(x5), errfun(x1, x5), errfun(x2, x5));
fprintf(' ProxGD Primal: cpu: %5.3f, iter: %5d, optval: %6.7E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t8, iter8, out8.fval, sparisity(x8), errfun_exact(x8), errfun(x1, x8), errfun(x2, x8));
fprintf('FProxGD Primal: cpu: %5.3f, iter: %5d, optval: %6.7E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t9, iter9, out9.fval, sparisity(x9), errfun_exact(x9), errfun(x1, x9), errfun(x2, x9));

plot_results(out2,out5,'Subgradient Descent',out8,'Proximal Gradient Descent',out9,'Fast Proximal Gradient Descent');
