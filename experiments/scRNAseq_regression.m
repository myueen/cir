clear
close all


data = readtable('.../pbmc_1_counts.csv');
data = data{:,2:size(data,2)};
data = data.';
cell_type = readtable('.../pbmc_1_cell_type.csv');
cell_type = cell_type{:,2};

% foregroun data
fg = data;
fg(any(ismissing(fg), 2), :) = [];

p = 100; % p can vary from 100 to 500
% select top p highly variable genes
col_var = var(fg,1);
[col_var_sorted,col_var_idx] = sort(-col_var);
fg = fg(:,col_var_idx(1:p));

% foreground label: cell types
Y = cell_type;
Y = categorical(Y);
Y = renamecats(Y,{'B cell';'CD4 T cell';'CD8 T cell';'NK cell';'Plasma cell';'cDC';'cMono';'ncMono';'pDC'},{'0','1','2','3','4','5','6','7','8'});
Y = str2double(string(Y));
labels = unique(Y); % set of unique foreground labels
L = length(labels); % number of foreground classes/slices
n = size(fg,1); % foreground sample size
X = fg-ones(n,1)*mean(fg,1);


bg =fg;
[m,p] = size(bg); % background sample size
Yt = 3*ones(m,1);% background labels
Yt = randi([1 9],m,1);

% tuning parameter
alpha = 1.5;

Ds = 2:1:10;
Accuracy_Tree = [];
Accuracy_KNN = [];
Accuracy_SVM = [];
Accuracy_boosting = [];
Accuracy_NN = [];


warning('off')
for i = 1:length(Ds)


    d = Ds(i);
    display(['d = ',num2str(d)])

    % PCA
    display('PCA......')
    [coeff,score,latent] = pca(X);
    X_PCA = score(:,1:d);

    % CPCA
    display('CPCA......')
    alpha_CPCA = 2; %
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

    %% 10 replicates, some classifier are slow that can be skipped
    % KNN is the most suitable classifier, fast, with high accuracy
    T = 10;
    for t = 1:T
        display(['t = ',num2str(t)])
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

        
%         display('Training SVM')
%         tic
%         % SVM
%         mdl_raw_SVM = fitcecoc(fg,Y,'CrossVal','on');
%         mdl_PCA_SVM = fitcecoc(X_PCA,Y,'CrossVal','on');
%         mdl_CPCA_SVM = fitcecoc(X_CPCA,Y,'CrossVal','on');
%         mdl_LDA_SVM = fitcecoc(X_LDA,Y,'CrossVal','on');
%         mdl_LASSO_SVM = fitcecoc(X_LASSO,Y,'CrossVal','on');
%         mdl_SIR_SVM  = fitcecoc(X_SIR,Y,'CrossVal','on');
%         mdl_CIR_SVM = fitcecoc(X_CIR,Y,'CrossVal','on');
%         Accuracy_raw_SVM = 1-kfoldLoss(mdl_raw_SVM);
%         Accuracy_PCA_SVM = 1-kfoldLoss(mdl_PCA_SVM);
%         Accuracy_CPCA_SVM = 1-kfoldLoss(mdl_CPCA_SVM);
%         Accuracy_LDA_SVM = 1-kfoldLoss(mdl_LDA_SVM);
%         Accuracy_LASSO_SVM = 1-kfoldLoss(mdl_LASSO_SVM);
%         Accuracy_SIR_SVM = 1-kfoldLoss(mdl_SIR_SVM);
%         Accuracy_CIR_SVM = 1-kfoldLoss(mdl_CIR_SVM);
%         Accuracy_SVM(i,:,t) = [Accuracy_raw_SVM,Accuracy_PCA_SVM,Accuracy_CPCA_SVM,Accuracy_LDA_SVM,Accuracy_LASSO_SVM,Accuracy_SIR_SVM,Accuracy_CIR_SVM];
%         toc

        display('Training Boosting')
        tic
        % Boosting
        mdl_raw_boosting = fitcensemble(fg,Y,'CrossVal','on');
        mdl_PCA_boosting = fitcensemble(X_PCA,Y,'CrossVal','on');
        mdl_CPCA_boosting = fitcensemble(X_CPCA,Y,'CrossVal','on');
        mdl_LDA_boosting = fitcensemble(X_LDA,Y,'CrossVal','on');
        mdl_LASSO_boosting = fitcensemble(X_LASSO,Y,'CrossVal','on');
        mdl_SIR_boosting  = fitcensemble(X_SIR,Y,'CrossVal','on');
        mdl_CIR_boosting = fitcensemble(X_CIR,Y,'CrossVal','on');
        Accuracy_raw_boosting = 1-kfoldLoss(mdl_raw_boosting);
        Accuracy_PCA_boosting = 1-kfoldLoss(mdl_PCA_boosting);
        Accuracy_CPCA_boosting = 1-kfoldLoss(mdl_CPCA_boosting);
        Accuracy_LDA_boosting = 1-kfoldLoss(mdl_LDA_boosting);
        Accuracy_LASSO_boosting = 1-kfoldLoss(mdl_LASSO_boosting);
        Accuracy_SIR_boosting = 1-kfoldLoss(mdl_SIR_boosting);
        Accuracy_CIR_boosting = 1-kfoldLoss(mdl_CIR_boosting);
        Accuracy_boosting(i,:,t) = [Accuracy_raw_boosting,Accuracy_PCA_boosting,Accuracy_CPCA_boosting,Accuracy_LDA_boosting,Accuracy_LASSO_boosting,Accuracy_SIR_boosting,Accuracy_CIR_boosting];
        toc

        display('Training Neural Network')
        tic
        % Neural Network
        mdl_raw_NN = fitcnet(fg,Y,'CrossVal','on');
        mdl_PCA_NN = fitcnet(X_PCA,Y,'CrossVal','on');
        mdl_CPCA_NN = fitcnet(X_CPCA,Y,'CrossVal','on');
        mdl_LDA_NN = fitcnet(X_LDA,Y,'CrossVal','on');
        mdl_LASSO_NN = fitcnet(X_LASSO,Y,'CrossVal','on');
        mdl_SIR_NN  = fitcnet(X_SIR,Y,'CrossVal','on');
        mdl_CIR_NN = fitcnet(X_CIR,Y,'CrossVal','on');
        Accuracy_raw_NN = 1-kfoldLoss(mdl_raw_NN);
        Accuracy_PCA_NN = 1-kfoldLoss(mdl_PCA_NN);
        Accuracy_CPCA_NN = 1-kfoldLoss(mdl_CPCA_NN);
        Accuracy_LDA_NN = 1-kfoldLoss(mdl_LDA_NN);
        Accuracy_LASSO_NN = 1-kfoldLoss(mdl_LASSO_NN);
        Accuracy_SIR_NN = 1-kfoldLoss(mdl_SIR_NN);
        Accuracy_CIR_NN = 1-kfoldLoss(mdl_CIR_NN);
        Accuracy_NN(i,:,t) = [Accuracy_raw_NN,Accuracy_PCA_NN,Accuracy_CPCA_NN,Accuracy_LDA_NN,Accuracy_LASSO_NN,Accuracy_SIR_NN,Accuracy_CIR_NN];
        toc


        %display(['                     raw, ','PCA, ','CPCA, ','LDA, ','LASSO, ','SIR, ','CIR.'])
        %display(['KNN accuracy: ',num2str(Accuracy_raw_KNN),', ', num2str(Accuracy_PCA_KNN),', ',num2str(Accuracy_CPCA_KNN),', ',num2str(Accuracy_LDA_KNN),', ',num2str(Accuracy_LASSO_KNN),', ',num2str(Accuracy_SIR_KNN),', ', num2str(Accuracy_CIR_KNN)])
        %display(['Decision Tree accuracy: ',num2str(Accuracy_raw_Tree),', ', num2str(Accuracy_PCA_Tree),', ',num2str(Accuracy_CPCA_Tree),', ',num2str(Accuracy_LDA_Tree),', ',num2str(Accuracy_LASSO_Tree),', ', num2str(Accuracy_SIR_Tree),', ', num2str(Accuracy_CIR_Tree)])
    end
end


% plot accuracy of KNN for different d
Accuracy_KNN_mean = mean(Accuracy_KNN,3);
figure
hold on
for j = 1:size(Accuracy_KNN,2)
    plot(Ds,Accuracy_KNN_mean(:,j),'LineWidth',2)
end
legend('Raw','PCA','CPCA','LDA','LASSO','SIR','CIR','FontSize',48)
hold off
title('KNN Accuracy')

% you may plot other classifiers that are sub-optimal
% Accuracy_Tree_mean = mean(Accuracy_Tree,3);
% figure
% hold on
% for j = 1:size(Accuracy_KNN,2)
%     plot(Ds,Accuracy_Tree_mean(:,j),'LineWidth',2)
% end
% legend('Raw','PCA','CPCA','LDA','LASSO','SIR','CIR','FontSize',48)
% hold off
% title('Tree Accuracy')


