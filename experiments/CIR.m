function V = CIR(fg,Y,bg,Yt,alpha,d)

n = size(fg,1); % foreground sample size
Y = double(Y); % foreground labels
labels = unique(Y); % set of unique foreground labels
L = length(labels); % number of foreground classes/slices

[m,p] = size(bg); % background sample size

labelst = unique(Yt); % set of unique background labels
Lt = length(labelst); % number of background classes/slices


X = fg - ones(n,1)*mean(fg,1); % shift foreground data zero mean
Xt = bg - ones(m,1)*mean(bg,1); % shift background to zero mean

Sigma_XX = X.'*X/n; % covaraince of foreground data
Sigma_XtXt = Xt.'*Xt/m; % covariance of background data


% estimate cov(E(X|Y))
Sigma_X = zeros(p,p);
for l = 1:L
    X_curr = fg(find(Y==labels(l)),:);
    n_curr = size(X_curr,1);
    Sigma_X = Sigma_X + (mean(X_curr,1)-mean(fg,1)).'*(mean(X_curr,1)-mean(fg,1))*n_curr;
end

Sigma_X = Sigma_X/n; % covariance of inverse curve E[X|Y]

if alpha == 0
    [coeff, D] = eig(Sigma_XX,Sigma_X);
    V = coeff(:,1:d);
else

    % estimate cov(E(\tilde{X}|\tilde{Y}))
    Sigma_Xt = zeros(p,p);
    for l = 1:Lt
        Xt_curr = bg(find(Yt==labelst(l)),:);
        nt_curr = size(Xt_curr,1);
        Sigma_Xt = Sigma_Xt + (mean(Xt_curr,1)-mean(bg,1)).'*(mean(Xt_curr,1)-mean(bg,1))*nt_curr;
    end

    Sigma_Xt = Sigma_Xt/m; % covariance of inverse curve E[X|Y]



    % calculate A, B, \tilde{A}, \tilde{B}
    A = Sigma_XX*Sigma_X*Sigma_XX;
    A(1,1:3)
    B = Sigma_XX*Sigma_XX;
    At = Sigma_XtXt*Sigma_Xt*Sigma_XtXt;
    Bt = Sigma_XtXt*Sigma_XtXt;



    % minimize the loss function by SGPM optimization algorithm
    V0 = rand(p,d);
    [V0,~] = qr(V0,0);
    V0 = eye(p);
    V0 = V0(:,1:d);


    %-------------------------------------------------------------------------
    % Solving the problem with our solver
    %-------------------------------------------------------------------------
    opts.gtol = 1e-5;
    opts.xtol = 1e-20;
    opts.ftol = 1e-20;
    opts.mxitr = 3000;

    %--- Our solver ---
    % Scaled Gradient Projection Method
    opts.alpha = 0.85;
    [V, out, F, Grad]= SGPM(V0, @fun, opts,A,B,At,Bt,alpha);
    out.fval = -2*out.fval; % convert the function value to the sum of eigenvalues

    %-------------------------------------------------------------------------
    % Printing the results
    %-------------------------------------------------------------------------
    fprintf('---------------------------------------------------\n')
    fprintf('Results for Scaled Gradient Projection Method \n')
    fprintf('---------------------------------------------------\n')
    fprintf('   Obj. function = %7.6e\n',  out.fval);
    fprintf('   Gradient norm = %7.6e \n', out.nrmG);
    fprintf('   ||X^T*X-I||_F = %3.2e\n',out.feasi )
    fprintf('   Iteration number = %d\n',  out.itr);
    fprintf('   Cpu time (secs) = %3.4f  \n',  out.time);
    fprintf('   Number of evaluation(Obj. func) = %d\n',  out.nfe);
    %-------------------------------------------------------------------------
end


    function [F, G] = fun(V,A,B,C,D,alpha)
        F = -trace(V.'*A*V*inv(V.'*B*V))+alpha*trace(V.'*C*V*inv(V.'*D*V));
        G = -2*A*V*inv(V.'*B*V)+2*B*V*inv(V.'*B*V)*V.'*A*V*inv(V.'*B*V)+2*alpha*C*V*inv(V.'*D*V)-2*alpha*D*V*inv(V.'*D*V)*V.'*C*V*inv(V.'*D*V);
    end

end