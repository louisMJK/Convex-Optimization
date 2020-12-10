% subgradient method for group lasso
function [x, iter, out] = gl_SGD_primal(x0, A, b, mu0, opts)
tic;
% default parameters
if ~isfield(opts, 'maxit');             opts.maxit = 100; end
if ~isfield(opts, 'maxit_inn');         opts.maxit_inn = 500; end
if ~isfield(opts, 'ftol');              opts.ftol = 1e-10; end
if ~isfield(opts, 'gtol');              opts.gtol = 1e-8; end
if ~isfield(opts, 'factor');            opts.factor = 0.1; end
if ~isfield(opts, 'mu1');               opts.mu1 = 100; end
if ~isfield(opts, 'gtol_init_ratio');   opts.gtol_init_ratio = 1/opts.gtol; end
if ~isfield(opts, 'ftol_init_ratio');   opts.ftol_init_ratio = 1e5; end
if ~isfield(opts, 'opts1');             opts.opts1 = struct(); end
if ~isfield(opts, 'etaf');              opts.etaf = 1e-1; end
if ~isfield(opts, 'etag');              opts.etag = 1e-1; end
L = eigs(A'*A,1);
if ~isfield(opts, 'alpha0');            opts.alpha0 = 1/L; end

% initialization
out = struct();
out.f = [];
itr = 0;
x = x0;
mu_t = opts.mu1;

f = Func(A, b, mu_t, x);

opts1 = opts.opts1;
opts1.ftol = opts.ftol*opts.ftol_init_ratio;
opts1.gtol = opts.gtol*opts.gtol_init_ratio;
out.itr_inn = 0;

while itr < opts.maxit
    opts1.maxit = opts.maxit_inn;
    opts1.gtol = max(opts1.gtol*opts.etag, opts.gtol);
    opts1.ftol = max(opts1.ftol*opts.etaf, opts.ftol);
    opts1.alpha0 = opts.alpha0;
    
    fp = f;
    % update f and X
    [x, out1] = gl_subgrad_inn(x, A, b, mu_t, mu0, opts1);
    f = out1.f(end);
    out.f = [out.f, out1.f];
    itr = itr + 1;
    
    norm_G = norm( A'*(A*x-b)+sub_h(x), 2 );

    % update mu
    if ~out1.flag
        mu_t = max(mu_t*opts.factor, mu0);
    end
    % stop
    if mu_t == mu0 && (norm_G < opts.gtol || abs(f-fp) < opts.ftol)
        break;
    end
    
    out.itr_inn = out.itr_inn + out1.itr;
end

out.fval = f;
out.itr = itr;
out.t = toc;
iter = out.itr_inn;


function f = Func(A, b, mu0, x)
    f = 0.5*norm(A*x-b,'fro')^2 + mu0*sum(norms(x,2,2));
end
% subgradient of ||X||1,2 
function [g] = sub_h(x)
    [n,l] = size(x);
    g = zeros(n,l);
    for i = 1:n
        if norm(x(i,1:l),2) ~= 0
            g(i,1:l) = x(i,1:l) / norm(x(i,1:l),2);
        else
            g(i,1:l) = 0.1*rand(1,l)/sqrt(l);
        end
    end
end
function [x, out] = gl_subgrad_inn(x, A, b, mu, mu0, opts)

if mu > mu0
    opts.step_type = 'fixed';
else
    opts.step_type = 'diminishing';
end

out = struct();
out.f = [];
r = A*x - b;
gx = A'*r;

sub_f = gx + mu*sub_h(x);
f_best = inf;

for k = 1:opts.maxit
    % update X
    alpha = set_step(k, opts);
    x = x - alpha * sub_f;
    x(abs(x) < 1e-4) = 0;
    
    % subgradient of objective function
    r = A*x - b;
    sub_f = A'*r + mu*sub_h(x);

    f_val = 0.5*norm(r,'fro')^2 + mu*sum(norms(x,2,2));
    out.f_hist(k) = f_val;
    
    % update optimal value
    f_best = min(f_best, f_val);
    out.f_hist_best(k) = f_best;
    out.f = [out.f, 0.5*norm(r,'fro')^2 + mu0*sum(norms(x,2,2))];
    
    % stop
    FDiff = abs( out.f_hist(k)-out.f_hist(max(k-1,1))) / abs(out.f_hist_best(1) );
    BFDiff = out.f_hist_best(max(k-10,1)) - out.f_hist_best(k);
    
    if (k > 1 && FDiff < opts.ftol) || (k > 10 && BFDiff < opts.ftol)
        break;
    end
end

if k == opts.maxit
    out.flag = 1;
else
    out.flag = 0;
end

out.itr = k;


function a = set_step(k, opts)
type = opts.step_type;
if strcmp(type, 'fixed')
    a = opts.alpha0;
elseif strcmp(type, 'diminishing')
    a = opts.alpha0 / (max(k,100)-99);
end

end
end
end
