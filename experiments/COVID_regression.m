clear
close all

fg = readtable('covid_preprocessed_fg.csv');
bg = readtable('covid_preprocessed_bg.csv');
Y = readtable('covid_preprocessed_Y.csv');
Yt = readtable('covid_preprocessed_Yt.csv');

fg = fg(2:end, 2:end);
bg = bg(2:end, 2:end);
Y = Y(:, 2:end);
Yt = Yt(:, 2:end);


fg = table2array(fg);
bg = table2array(bg);

Y = table2array(Y);
Yt = table2array(Yt);


% Set the seed for reproducibility
seed = 42;
rng(seed);



labels = unique(Y); % set of unique foreground labels
L = length(labels); % number of foreground classes/slices
n = size(fg,1); % foreground sample size
X = fg-ones(n,1)*mean(fg,1);


labelst = unique(Yt); % set of unique background labels
Lt = length(labelst); % number of background classes/slices
[m,p] = size(bg); % background sample size




opt_alpha = [1e-5, 1e-8, 1e-6, 1e-8, 1e-09, 1e-7];
Ds = 2:1:7;
Accuracy_Tree = [];
Accuracy_KNN = [];

for i = 1:length(Ds)

    d = Ds(i);
    alpha = opt_alpha(i);


    % PCA
    display('PCA......')
    [coeff,score,latent] = pca(X);
    X_PCA = score(:,1:d);


    % CPCA
    display('CPCA......')
    alpha_CPCA = 2; 
    [coeff_CPCA,D_CPCA] = eig(cov(fg)-alpha_CPCA*cov(bg));
    V_CPCA = coeff_CPCA(:,1:d);
    X_CPCA = X*V_CPCA;


    % LDA
    display('LDA......')
    [X_LDA,V_LDA,lambda_LDA] = LDA(X,Y);
    X_LDA = X*V_LDA(:,1:d);

    % LASSO
    display('LASSO......')
    B = lasso(X,Y);
    B_sum = sum(B~=0,1);
    lambda_idx = min(find(B_sum<=d));
    selected_idx = find(B(:,lambda_idx)~=0);
    X_LASSO = X(:,selected_idx);


    display('SIR......')
    % CIR code from Li 1991 AoS
    Sigma_XX = X.'*X/n;
    % estimate cov(E(X|Y))
    Sigma_X = zeros(p,p);
    for l = 1:L
        X_curr = fg(find(Y==labels(l)),:);
        n_curr = size(X_curr,1);
        Sigma_X = Sigma_X + (mean(X_curr,1)-mean(fg,1)).'*(mean(X_curr,1)-mean(fg,1))*n_curr;
    end

    Sigma_X = Sigma_X/n; % covariance of inverse curve E[X|Y]

    [coeff_SIR, D_SIR] = eig(Sigma_XX,Sigma_X);
    V_SIR = coeff_SIR(:,1:d);
    X_SIR = X*V_SIR;


    % CIR
    display('CIR......')
    V_CIR = CIR(fg,Y,bg,Yt,alpha,d);
    X_CIR = X*V_CIR;


    %% Multiple classifiers, KNN is the most suitable one here
    T = 10;
    for t = 1:T

        display('Training KNN')
        tic
        % KNN
        mdl_raw_KNN = fitcknn(fg,Y,'CrossVal','on');
        mdl_PCA_KNN = fitcknn(X_PCA,Y,'CrossVal','on');
        mdl_CPCA_KNN = fitcknn(X_CPCA,Y,'CrossVal','on');
        mdl_LDA_KNN = fitcknn(X_LDA,Y,'CrossVal','on');
        mdl_LASSO_KNN = fitcknn(X_LASSO,Y,'CrossVal','on');
        mdl_SIR_KNN  = fitcknn(X_SIR,Y,'CrossVal','on');
        mdl_CIR_KNN = fitcknn(X_CIR,Y,'CrossVal','on');
        Accuracy_raw_KNN = 1-kfoldLoss(mdl_raw_KNN);
        Accuracy_PCA_KNN = 1-kfoldLoss(mdl_PCA_KNN);
        Accuracy_CPCA_KNN = 1-kfoldLoss(mdl_CPCA_KNN);
        Accuracy_LDA_KNN = 1-kfoldLoss(mdl_LDA_KNN);
        Accuracy_LASSO_KNN = 1-kfoldLoss(mdl_LASSO_KNN);
        Accuracy_SIR_KNN = 1-kfoldLoss(mdl_SIR_KNN);
        Accuracy_CIR_KNN = 1-kfoldLoss(mdl_CIR_KNN);
        Accuracy_KNN(i,:,t) = [Accuracy_raw_KNN,Accuracy_PCA_KNN,Accuracy_CPCA_KNN,Accuracy_LDA_KNN,Accuracy_LASSO_KNN,Accuracy_SIR_KNN,Accuracy_CIR_KNN];
        toc


        display('Training Decision Tree')
        tic
        % Decision Tree
        mdl_raw_Tree = fitctree(fg,Y,'CrossVal','on');
        mdl_PCA_Tree = fitctree(X_PCA,Y,'CrossVal','on');
        mdl_CPCA_Tree = fitctree(X_CPCA,Y,'CrossVal','on');
        mdl_LDA_Tree = fitctree(X_LDA,Y,'CrossVal','on');
        mdl_LASSO_Tree = fitctree(X_LASSO,Y,'CrossVal','on');
        mdl_SIR_Tree  = fitctree(X_SIR,Y,'CrossVal','on');
        mdl_CIR_Tree = fitctree(X_CIR,Y,'CrossVal','on');
        Accuracy_raw_Tree = 1-kfoldLoss(mdl_raw_Tree);
        Accuracy_PCA_Tree = 1-kfoldLoss(mdl_PCA_Tree);
        Accuracy_CPCA_Tree = 1-kfoldLoss(mdl_CPCA_Tree);
        Accuracy_LDA_Tree = 1-kfoldLoss(mdl_LDA_Tree);
        Accuracy_LASSO_Tree = 1-kfoldLoss(mdl_LASSO_Tree);
        Accuracy_SIR_Tree = 1-kfoldLoss(mdl_SIR_Tree);
        Accuracy_CIR_Tree = 1-kfoldLoss(mdl_CIR_Tree);
        Accuracy_Tree(i,:,t) = [Accuracy_raw_Tree,Accuracy_PCA_Tree,Accuracy_CPCA_Tree,Accuracy_LDA_Tree,Accuracy_LASSO_Tree,Accuracy_SIR_Tree,Accuracy_CIR_Tree];
        toc

    

       display(['raw, ','PCA, ','CPCA, ','LDA, ','LASSO, ','SIR, ','CIR.'])
    

       display(['KNN accuracy: ',num2str(Accuracy_raw_KNN),', ', num2str(Accuracy_PCA_KNN),', ',num2str(Accuracy_CPCA_KNN),', ',num2str(Accuracy_LDA_KNN),', ',num2str(Accuracy_LASSO_KNN),', ',num2str(Accuracy_SIR_KNN),', ', num2str(Accuracy_CIR_KNN)])
       display(['Decision Tree accuracy: ',num2str(Accuracy_raw_Tree),', ', num2str(Accuracy_PCA_Tree),', ',num2str(Accuracy_CPCA_Tree),', ',num2str(Accuracy_LDA_Tree),', ',num2str(Accuracy_LASSO_Tree),', ', num2str(Accuracy_SIR_Tree),', ', num2str(Accuracy_CIR_Tree)])
  
    end
end



Accuracy_KNN_mean = mean(Accuracy_KNN,3);
Accuracy_Tree_mean = mean(Accuracy_Tree, 3);
disp('KNN Accuracy')
disp('..........................................')
disp(['2, ','3, ','4, ','5, ','6, ','7.'])
disp(Accuracy_KNN_mean')

disp('Tree Accuracy')
disp('..........................................')
disp(['2, ','3, ','4, ','5, ','6, ','7.'])
disp(Accuracy_Tree_mean')