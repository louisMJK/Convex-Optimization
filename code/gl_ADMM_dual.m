% ADMM for the dual problem
function [X, iter, out] = gl_ADMM_dual(x0, A, B, mu, opts)
% default parameters
if ~isfield(opts, 'maxit'); opts.maxit = 5000; end
if ~isfield(opts, 'ftol');  opts.ftol = 1e-10; end
if ~isfield(opts, 'gtol');  opts.gtol = 1e-8; end
if ~isfield(opts, 'rho');   opts.rho = 0.01; end
if ~isfield(opts, 'gamma'); opts.gamma = 1.618; end

% initialization
out = struct();
out.f = [];
out.f_best = [];
k = 0;

[m,n] = size(A);
l = size(x0,2);
Z = randn(m,l);
W = zeros(n,l);
X = zeros(n,l);
rho = opts.rho;
gamma = opts.gamma;

fp = inf;
f_best = inf;
normC = inf;
f = Func(A, B, mu, X);

while k < opts.maxit && abs(f-fp) > opts.ftol && normC > opts.gtol
    fp = f;
    
    % iterate
    W_p = W;
    W = Proj(Z, X, A, mu, rho);
    Z = ( eye(m) + rho*(A*A') ) \ (A*(X-rho*W) - B);
    X = X - gamma*rho*(W+A'*Z);
    X(abs(X)<1e-4) = 0;
    rho = set_rho(rho, Z, W_p, W, A);
    
    % objective value
    f = Func(A, B, mu, X);
    out.f = [out.f, f];
    C = W + A'*Z;
    normC = norm(C,'fro');
    
    % update optimal value
    f_best = min(f_best, f);
    out.f_best = [out.f_best, f_best];
    
    k = k + 1;
end

out.fval = f;
out.itr = k;
out.Z = Z;
out.W = W;
iter = k;
end

function f = Func(A, B, mu, X)
    f = 0.5*norm(A*X-B,'fro')^2 + mu*sum(norms(X,2,2));
end
function W = Proj(Z, X, A, mu, rho)
W = (1/rho)*X - A'*Z;
for i = 1:size(W,1)
    if norm(W(i,:)) > mu
        W(i,:) = (mu/norm(W(i,:))) * W(i,:);
    end
end
end
function rho = set_rho(rho_p, Z, W_p, W, A)
r = norm(W+A'*Z, 'fro');
s = norm(A*(W_p-W), 'fro');
if r > 10*s
    rho = rho_p*2;
elseif s > 10*r
    rho = rho_p/2;
else
    rho = rho_p;
end
end