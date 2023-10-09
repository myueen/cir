function [X, out, F_eval, Grad]= SGPM(X, fun, opts, varargin)
%-------------------------------------------------------------------------
% A Scaled Gradient Projection Method for Minimization over the Stiefel Manifold
%
%   min F(X), S.t., X'*X = I_p, where X \in R^{n,p}
%
%-------------------------------------------------------------------------
% Input:
%           X --- n by p matrix such that X'*X = I
%         fun --- objective function and its gradient:
%                 [F, G] = fun(X,  data1, data2)
%                 F, G are the objective function value and gradient, repectively
%                 data1, data2 are addtional data, and can be more
%                 Calling syntax:
%                   [X, out]= OptStiefelGBB(X0, @fun, opts, data1, data2);
%
%        opts --- option structure with fields:
%                 record = 0, no print out
%                 mxitr       max number of iterations
%                 xtol        stop control for ||X_k - X_{k-1}||
%                 gtol        stop control for the projected gradient
%                 ftol        stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
%                             usually, max{xtol, gtol} > ftol
%                 alpha       the parameter 0<="alpha"<=1 corrsponding to
%                             the initial scaling parameter
% Output:
%           X --- solution
%         out --- output information
%      F_eval --- evaluation of the objective function in each iteration
%        Grad --- norm of gradient of lagrangean function espect to primal 
%                 variables in each iteration
%-------------------------------------------------------------------------
%
% Reference: 
% H. Oviedo and O. Dalmau
% A Scaled Gradient Projection Method for Minimization over the Stiefel Manifold
% DOI: https://doi.org/10.1007/978-3-030-33749-0_20
%
% Authors:
% Harry F. Oviedo and Oscar S. Dalmau 
%     Harry Oviedo <harry.oviedo@cimat.mx>
%     Oscar Dalmau <dalmau@cimat.mx>
% Version 1.0 .... 2019/12
%-------------------------------------------------------------------------


%% Size information
if isempty(X)
    error('input X is an empty matrix');
else
    [n, k] = size(X);
end

if isfield(opts, 'xtol')
    if opts.xtol < 0 || opts.xtol > 1
        opts.xtol = 1e-6;
    end
else
    opts.xtol = 1e-6;
end

if isfield(opts, 'gtol')
    if opts.gtol < 0 || opts.gtol > 1
        opts.gtol = 1e-4;
    end
else
    opts.gtol = 1e-4;
end

if isfield(opts, 'ftol')
    if opts.ftol < 0 || opts.ftol > 1
        opts.ftol = 1e-12;
    end
else
    opts.ftol = 1e-12;
end

% parameters for control the linear approximation in line search
if isfield(opts, 'rho')
   if opts.rho < 0 || opts.rho > 1
        opts.rho = 1e-4;
   end
else
    opts.rho = 1e-4;
end

% factor for decreasing the step size in the backtracking line search
if isfield(opts, 'eta')
   if opts.eta < 0 || opts.eta > 1
        opts.eta = 0.1;
   end
else
    opts.eta = 0.2;
end

% parameters for updating C by HongChao, Zhang
if isfield(opts, 'gamma')
   if opts.gamma < 0 || opts.gamma > 1
        opts.gamma = 0.85;
   end
else
    opts.gamma = 0.85;
end

if isfield(opts, 'tau')
   if opts.tau < 0 || opts.tau > 1e3
        opts.tau = 1e-3;
   end
else
    opts.tau = 1e-3;
end

% parameters for the  nonmontone line search by Raydan
if ~isfield(opts, 'STPEPS')
    opts.STPEPS = 1e-10;
end

if isfield(opts, 'nt')
    if opts.nt < 0 || opts.nt > 100
        opts.nt = 5;
    end
else
    opts.nt = 5;
end

if isfield(opts, 'projG')
    switch opts.projG
        case {1,2}; otherwise; opts.projG = 1;
    end
else
    opts.projG = 1;
end

if isfield(opts, 'iscomplex')
    switch opts.iscomplex
        case {0, 1}; otherwise; opts.iscomplex = 0;
    end
else
    opts.iscomplex = 0;
end

if isfield(opts, 'mxitr')
    if opts.mxitr < 0 || opts.mxitr > 2^20
        opts.mxitr = 1000;
    end
else
    opts.mxitr = 1000;
end

if isfield(opts, 'alpha')
    if opts.alpha < 0 || opts.alpha > 1
        opts.alpha = 0;
    end
else
    opts.alpha = 0.85;
end

if ~isfield(opts, 'record')
    opts.record = 0;
end


%-------------------------------------------------------------------------------
% copy parameters
xtol = opts.xtol;
gtol = opts.gtol;
ftol = opts.ftol;
rho  = opts.rho;
alpha = opts.alpha;
eta   = opts.eta;
gamma = opts.gamma;
nt = opts.nt;   crit = ones(nt, 3);

invH = true; if k < n/2; invH = false;  eye2k = eye(2*k); end
%% Initial function value and gradient
% prepare for iterations
[F,  G] = feval(fun, X , varargin{:});  out.nfe = 1;  
GX = G'*X;

if invH
    GXT = G*X';  H = (GXT - GXT');    
else
    if opts.projG == 1
        U =  [G, X];    V = [X, -G];       VU = V'*U;
    elseif opts.projG == 2
        GB = G - 0.5*X*(X'*G);
        U =  [GB, X];    V = [X, -GB];       VU = V'*U;
    end
    VX = V'*X;
end
dtX = G - X*GX;     nrmG  = norm(dtX, 'fro');  
Q = 1; Cval = F;  tau = opts.tau;

%% Print iteration header if debug == 1
if (opts.record == 1)
    fid = 1;
    fprintf(fid, '----------- Scaled Gradient Projection Method with Line search ----------- \n');
    fprintf(fid, '%4s %8s %8s %10s %10s\n', 'Iter', 'tau', 'F(X)', 'nrmG', 'XDiff');
end

%% main iteration
F_eval = zeros(opts.mxitr+1,1 );
Grad = zeros(opts.mxitr+1,1 );
F_eval(1) = F;
Grad(1) = nrmG; 

tstart = tic;
for itr = 1 : opts.mxitr
    XP = X;     FP = F;   dtXP = dtX;   nrmGP = nrmG;
    
    % scale step size
    nls = 1; deriv = rho*nrmG^2; 
    
    while 1
        % Update Scheme
        if invH
            
            if abs(alpha) < rho               % Explicit Euler (Steepest Descent)
                X = XP - tau*dtXP;
            elseif abs(alpha - 0.5) < rho         % Crank-Nicolson
                X = linsolve(eye(n) + (tau*0.5)*H, XP - (0.5*tau)*dtXP);
            elseif abs(alpha - 1) < rho           % Implicit Euler 
                X = linsolve(eye(n) + tau*H, XP);
            else                        % Convex Combination
                X = linsolve(eye(n) + (tau*alpha)*H, XP - ((1-alpha)*tau)*dtXP);
            end
            
            if abs(alpha - 0.5) > rho
                XtX = X'*X;
                L = chol(XtX);     X = X/L;
            end       
        else
            [aa, ~] = linsolve(eye2k + (alpha*tau)*VU, VX);
            X = XP - U*(tau*aa);
            
            if abs(alpha - 0.5) > rho
                XtX = X'*X;
                L = chol(XtX);     X = X/L;
            end
            
        end
        
        % calculate G, F
        [F,G] = feval(fun, X, varargin{:});
        out.nfe = out.nfe + 1;
        
        if F <= Cval - tau*deriv || nls >= 5
            break;
        end
        tau = eta*tau;          nls = nls + 1;
    end  
    
    GX = G'*X;
    dtX = G - X*GX;         nrmG  = norm(dtX, 'fro');
    F_eval(itr+1) = F;      Grad(itr+1) = nrmG;
    
    % Adaptive scaling matrix strategy
    if nrmG < nrmGP
        if nrmG >=  0.5*nrmGP
            alpha = max(min(alpha*1.1,1),0);
        end
    else
        alpha = max(min(alpha*0.9,0),0.5);
    end
    
    % Computing the Riemannian Gradient
    if invH
        if abs(alpha) > rho
            GXT = G*X';  H = GXT - GXT';    
        end        
    else
        if opts.projG == 1
            U =  [G, X];    V = [X, -G];       VU = V'*U;
        elseif opts.projG == 2
            GB = G - X*(0.5*GX');
            U  =  [GB, X];    V = [X, -GB];     VU = V'*U; 
        end
        VX = V'*X;
    end
    
    % Compute the Alternate ODH step-size:
    S = X - XP;             SS = sum(sum(S.*S));
    XDiff = sqrt(SS/n);     FDiff = abs(FP-F)/(abs(FP)+1);
    
    Y = dtX - dtXP;     SY = abs(sum(sum(S.*Y)));
        
       
    if mod(itr,2)==0 
        tau = SS/SY;
    else
        YY = sum(sum(Y.*Y)); 
        tau  = SY/YY;
    end
    tau = max(min(tau, 1e20), 1e-20);
    
    % Stopping Rules
    crit(itr,:) = [nrmG, XDiff, FDiff];
    mcrit = mean(crit(itr-min(nt,itr)+1:itr, :),1);   
    if ( XDiff < xtol && FDiff < ftol ) || nrmG < gtol || all(mcrit(2:3) < 10*[xtol, ftol])  
        if itr <= 2
            ftol = 0.1*ftol;
            xtol = 0.1*xtol;
            gtol = 0.1*gtol;
        else
            out.msg = 'converge';
            break;
        end
    end
    Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + F)/Q;
 end
tiempo = toc(tstart);

F_eval = F_eval(1:itr,1);
Grad = Grad(1:itr,1);
if itr >= opts.mxitr
    out.msg = 'exceed max iteration';
end

out.feasi = norm(X'*X-eye(k),'fro');
if  out.feasi > 1e-13
    L = chol(X'*X);     
    X = X/L;
    [F,G] = feval(fun, X, varargin{:});
    dtX = G - X*GX;         
    nrmG  = norm(dtX, 'fro');
    out.nfe = out.nfe + 1;
    out.feasi = norm(X'*X-eye(k),'fro');
end

out.feasi = norm(X'*X-eye(k),'fro');
out.nrmG = nrmG;
out.fval = F;
out.itr = itr;
out.time = tiempo;
end
