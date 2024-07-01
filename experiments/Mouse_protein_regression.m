clear
close all

data = readtable('.../Data_Cortex_Nuclear.csv');

% foreground data
fg = data;
fg(any(ismissing(fg), 2), :) = [];

% foreground label
Y = fg.class;
Y = categorical(Y);
Y = renamecats(Y,{'c-CS-m','c-CS-s','c-SC-m','c-SC-s','t-CS-m','t-CS-s','t-SC-m','t-SC-s'},{'0','1','2','3','4','5','6','7'});
Y = str2double(string(Y));
labels = unique(Y); % set of unique foreground labels

% foreground slices
L = length(labels); % number of foreground classes/slices
fg = fg{:,2:78};
n = size(fg,1); % foreground sample size
X = fg-ones(n,1)*mean(fg,1);


% background data
bg = data(strcmp(data.Genotype,'Control'),:);
bg(any(ismissing(bg), 2), :) = [];
% background label
Yt = bg.Behavior;
Yt = categorical(Yt);
Yt = renamecats(Yt,{'C/S','S/C'},{'0','1'});
Yt = str2double(string(Yt));
labelst = unique(Yt); % set of unique background labels
Lt = length(labelst); % number of background classes/slices
bg = bg{:,2:78};
[m,p] = size(bg); % background sample size

% tuning parameter
opt_alpha = [0.0005,0.0005,0.0002,0.00001,0.0000001,0.000001,0.0000001,0.0000001,0.0000001];
Ds = 2:1:7;
Accuracy_Tree = [];
Accuracy_KNN = [];
Accuracy_SVM = [];
Accuracy_boosting = [];
Accuracy_NN = [];
for i = 1:length(Ds)


    d = Ds(i);
    alpha = opt_alpha(i);
    alpha = 0.0002;

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
           




        display('Training SVM')
        tic
        % SVM
        mdl_raw_SVM = fitcnet(fg,Y,'CrossVal','on');
        mdl_PCA_SVM = fitcnet(X_PCA,Y,'CrossVal','on');
        mdl_CPCA_SVM = fitcnet(X_CPCA,Y,'CrossVal','on');
        mdl_LDA_SVM = fitcnet(X_LDA,Y,'CrossVal','on');
        mdl_LASSO_SVM = fitcnet(X_LASSO,Y,'CrossVal','on');
        mdl_SIR_SVM  = fitcnet(X_SIR,Y,'CrossVal','on');
        mdl_CIR_SVM = fitcnet(X_CIR,Y,'CrossVal','on');
        Accuracy_raw_SVM = 1-kfoldLoss(mdl_raw_SVM);
        Accuracy_PCA_SVM = 1-kfoldLoss(mdl_PCA_SVM);
        Accuracy_CPCA_SVM = 1-kfoldLoss(mdl_CPCA_SVM);
        Accuracy_LDA_SVM = 1-kfoldLoss(mdl_LDA_SVM);
        Accuracy_LASSO_SVM = 1-kfoldLoss(mdl_LASSO_SVM);
        Accuracy_SIR_SVM = 1-kfoldLoss(mdl_SIR_SVM);
        Accuracy_CIR_SVM = 1-kfoldLoss(mdl_CIR_SVM);
        Accuracy_SVM(i,:,t) = [Accuracy_raw_SVM,Accuracy_PCA_SVM,Accuracy_CPCA_SVM,Accuracy_LDA_SVM,Accuracy_LASSO_SVM,Accuracy_SIR_SVM,Accuracy_CIR_SVM];
        toc

        display('Training Boosting')
        tic
        % Boosting
        mdl_raw_boosting = fitcecoc(fg,Y,'CrossVal','on');
        mdl_PCA_boosting = fitcecoc(X_PCA,Y,'CrossVal','on');
        mdl_CPCA_boosting = fitcecoc(X_CPCA,Y,'CrossVal','on');
        mdl_LDA_boosting = fitcecoc(X_LDA,Y,'CrossVal','on');
        mdl_LASSO_boosting = fitcecoc(X_LASSO,Y,'CrossVal','on');
        mdl_SIR_boosting  = fitcecoc(X_SIR,Y,'CrossVal','on');
        mdl_CIR_boosting = fitcecoc(X_CIR,Y,'CrossVal','on');
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
        mdl_raw_NN = fitcensemble(fg,Y,'CrossVal','on');
        mdl_PCA_NN = fitcensemble(X_PCA,Y,'CrossVal','on');
        mdl_CPCA_NN = fitcensemble(X_CPCA,Y,'CrossVal','on');
        mdl_LDA_NN = fitcensemble(X_LDA,Y,'CrossVal','on');
        mdl_LASSO_NN = fitcensemble(X_LASSO,Y,'CrossVal','on');
        mdl_SIR_NN  = fitcensemble(X_SIR,Y,'CrossVal','on');
        mdl_CIR_NN = fitcensemble(X_CIR,Y,'CrossVal','on');
        Accuracy_raw_NN = 1-kfoldLoss(mdl_raw_NN);
        Accuracy_PCA_NN = 1-kfoldLoss(mdl_PCA_NN);
        Accuracy_CPCA_NN = 1-kfoldLoss(mdl_CPCA_NN);
        Accuracy_LDA_NN = 1-kfoldLoss(mdl_LDA_NN);
        Accuracy_LASSO_NN = 1-kfoldLoss(mdl_LASSO_NN);
        Accuracy_SIR_NN = 1-kfoldLoss(mdl_SIR_NN);
        Accuracy_CIR_NN = 1-kfoldLoss(mdl_CIR_NN);
        Accuracy_NN(i,:,t) = [Accuracy_raw_NN,Accuracy_PCA_NN,Accuracy_CPCA_NN,Accuracy_LDA_NN,Accuracy_LASSO_NN,Accuracy_SIR_NN,Accuracy_CIR_NN];
        toc


       % display(['                     raw, ','PCA, ','CPCA, ','LDA, ','LASSO, ','SIR, ','CIR.'])
        %display(['KNN accuracy: ',num2str(Accuracy_raw_KNN),', ', num2str(Accuracy_PCA_KNN),', ',num2str(Accuracy_CPCA_KNN),', ',num2str(Accuracy_LDA_KNN),', ',num2str(Accuracy_LASSO_KNN),', ',num2str(Accuracy_SIR_KNN),', ', num2str(Accuracy_CIR_KNN)])
        %display(['Decision Tree accuracy: ',num2str(Accuracy_raw_Tree),', ', num2str(Accuracy_PCA_Tree),', ',num2str(Accuracy_CPCA_Tree),', ',num2str(Accuracy_LDA_Tree),', ',num2str(Accuracy_LASSO_Tree),', ', num2str(Accuracy_SIR_Tree),', ', num2str(Accuracy_CIR_Tree)])
    end
end



Accuracy_KNN_mean = mean(Accuracy_KNN,3);
figure
hold on
for j = 1:size(Accuracy_KNN,2)
    plot(Ds(1:6),Accuracy_KNN_mean(1:6,j),'LineWidth',2)
end
legend('Raw','PCA','CPCA','LDA','LASSO','SIR','CIR','FontSize',48)
hold off
title('KNN Accuracy','FontSize',48)


% present KNN accuracy here, with the best performance
% Accuracy_Tree_mean = mean(Accuracy_Tree,3);
% figure
% hold on
% for j = 1:size(Accuracy_KNN,2)
%     plot(Ds(1:6),Accuracy_Tree_mean(1:6,j),'LineWidth',2)
% end
% legend('Raw','PCA','CPCA','LDA','LASSO','SIR','CIR','FontSize',48)
% hold off
% title('Tree Accuracy')

matrix2latex(mean(Accuracy_KNN,3).','Mouse_Accuracy_KNN');
matrix2latex(mean(Accuracy_Tree,3).','Mouse_Accuracy_Tree');
matrix2latex(mean(Accuracy_SVM,3).','Mouse_Accuracy_SVM');
matrix2latex(mean(Accuracy_boosting,3).','Mouse_Accuracy_Boosting');
matrix2latex(mean(Accuracy_NN,3).','Mouse_Accuracy_NN');


matrix2latex(std(Accuracy_KNN,0,3).','Mouse_Accuracy_std_KNN');
matrix2latex(std(Accuracy_Tree,0,3).','Mouse_Accuracy_std_Tree');
matrix2latex(std(Accuracy_SVM,0,3).','Mouse_Accuracy_std_SVM');
matrix2latex(std(Accuracy_boosting,0,3).','Mouse_Accuracy_std_Boosting');
matrix2latex(std(Accuracy_NN,0,3).','Mouse_Accuracy_std_NN');

