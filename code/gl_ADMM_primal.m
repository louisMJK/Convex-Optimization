% ADMM with linearization for the primal problem
function [X, iter, out] = gl_ADMM_primal(x0, A, B, mu, opts)
% default parameters
if ~isfield(opts, 'maxit'); opts.maxit = 5000; end
if ~isfield(opts, 'ftol');  opts.ftol = 1e-8; end
if ~isfield(opts, 'gtol');  opts.gtol = 1e-10; end
if ~isfield(opts, 'rho');   opts.rho = 0.01; end
if ~isfield(opts, 'gamma'); opts.gamma = 1.618; end
L = eigs(A'*A,1);
if ~isfield(opts, 'alpha0');opts.alpha0 = 1/L; end

% initialization
out = struct();
out.f = [];
out.f_best = [];
k = 0;

X = x0;
[n,l] = size(X);
Y = zeros(n,l);
Z = zeros(n,l);
rho = opts.rho;
gamma = opts.gamma;

fp = inf;
f_best = inf;
normC = inf;
f = Func(A, B, mu, X);

while k < opts.maxit && abs(f-fp) > opts.ftol && normC > opts.gtol
    fp = f;
    
    % iterate
    t = opts.alpha0 / (max(k,100)-99);
%     t = opts.alpha0;
    X = X - t * ( ((A'*A)+rho*eye(n))*X - A'*B + Y - rho*Z );
    X(abs(X)<1e-4) = 0;
    C = Z - t*rho*(Z-X-Y/rho);
    Z = prox(C,mu,t);
    Y = Y + gamma*rho*(X-Z);
    
    % objective value
    f = Func(A, B, mu, X);
    out.f = [out.f, f];
    normC = norm(X-Z,'fro');
    
    % update optimal value
    f_best = min(f_best, f);
    out.f_best = [out.f_best, f_best];
    
    k = k + 1;
end

out.fval = f;
out.itr = k;
out.Y = Y;
iter = k;
end

% proximal operator of mu*||Z||1,2
function prox = prox(Z,mu,t)
prox = Z;
for i = 1:size(Z,1)
    if norm(Z(i,:)) <= t*mu
        prox(i,:) = 0;
    else
        prox(i,:) = Z(i,:) / norm(Z(i,:)) * (norm(Z(i,:))-t*mu);
    end
end
end
function f = Func(A, B, mu, X)
    f = 0.5*norm(A*X-B,'fro')^2 + mu*sum(norms(X,2,2));
end
% subgradient of ||X||1,2 
function g = SGD(Z)
    [n,l] = size(Z);
    g = zeros(n,l);
    for i = 1:n
        if norm(Z(i,:),2) ~= 0
            g(i,:) = Z(i,:) / norm(Z(i,:),2);
        else
            g(i,:) = 0.1*ones(1,l)/sqrt(l);
        end
    end
end