function [x, iter, out] = gl_GD_primal(x0, A, b, mu0, opts)
% default parameters
% smmothing parameter
if ~isfield(opts, 'gamma');             opts.gamma = 1e-4; end
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
gamma = opts.gamma;

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
    [x, out1] = gl_smoothing_inn(x, A, b, mu_t, mu0, opts1, gamma);
    f = out1.f(end);
    out.f = [out.f, out1.f];
    itr = itr + 1;
    
    norm_G = norm(A'*(A*x-b) + GD_h(x, gamma), 2);

    % update mu
    if ~out1.flag
        mu_t = max(mu_t * opts.factor, mu0);
    end
    % stop
    if mu_t == mu0 && (norm_G < opts.gtol || abs(f-fp) < opts.ftol)
        break;
    end
    out.itr_inn = out.itr_inn + out1.itr;
end

out.fval = f;
out.itr = itr;
iter = out.itr_inn;


function f = Func(A, b, mu0, x)
    f = 0.5*norm(A*x-b,'fro')^2 + mu0*sum(norms(x,2,2));
end
% gradient of smooth ||X||1,2 
function [g] = GD_h(x, gamma)
    [n,l] = size(x);
    g = zeros(n,l);
    for i = 1:n
        if norm(x(i,1:l),2) >= gamma
            g(i,1:l) = x(i,1:l) / norm(x(i,1:l),2);
        else
            g(i,1:l) = x(i,1:l) / gamma;
        end
    end
end
function [x, out] = gl_smoothing_inn(x, A, b, mu, mu0, opts, gamma)
% step size for mu_t
if mu > mu0
    opts.step_type = 'fixed';
else
    opts.step_type = 'diminishing';
end

out = struct();
out.f = [];
r = A*x - b;
gx = A'*r;

GD_f = gx + mu * GD_h(x,gamma);
f_best = inf;

for k = 1:opts.maxit
    % update X
    alpha = set_step(k, opts);
    x = x - alpha * GD_f;
    x(abs(x) < 1e-4) = 0;
    
    % gradient of objective function
    r = A*x - b;
    GD_f = A'*r + mu*GD_h(x,gamma);

    f_val = Func(A, b, mu, x);
    out.f_hist(k) = f_val;
    
    % update optimal value
    f_best = min(f_best, f_val);
    out.f_hist_best(k) = f_best;
    out.f = [out.f, Func(A, b, mu0, x)];
    
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