clear
close all


data = readtable('.../Retinol.txt');
data = data{:,:};



% foreground data
fg = data(:,1:12);
fg(any(ismissing(fg), 2), :) = [];
% foreground response (continuous)
Y = data(:,13);
[n,p] = size(fg); % foreground sample size
X = fg-ones(n,1)*mean(fg,1);
% foreground slices
L = 4;
partition = linspace(min(Y),max(Y),L);
labels = zeros(n,1);
for i = 1:n
    labels(i)=max(find((Y(i)>=partition)==1));
end

% background data
bg = data(:,1:12);
[m,p] = size(bg); % background sample size

% background response (continuous)
Yt = data(:,14);
Lt = 4;
Lt = L;
partitiont = linspace(min(Yt),max(Yt),L);
labelst = zeros(m,1);
for i = 1:m
    labelst(i)=max(find((Yt(i)>=partitiont)==1));
end



% reduced dimensions
Ds = 1:1:8;

% tuning parameter alpha
alpha_opt = zeros(length(Ds),1);
alpha_opt(1) = 0.1;
alpha_opt(2) = 0.1;
for i = 1:length(Ds)


    d = Ds(i);

    alpha = alpha_opt(i);

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
    lambda_idx = max(find(B_sum>d));
    selected_idx = find(B(:,lambda_idx)~=0);
    [selected_idx,idx]= sort(-selected_idx);
    X_LASSO = X(:,-selected_idx(1:d));


    display('SIR......')
    % CIR code from Li 1991 AoS
    Sigma_XX = X.'*X/n;
    % estimate cov(E(X|Y))
    L = 4;
    Sigma_X = zeros(p,p);
    for l = 1:L
        X_curr = fg(find(labels==l),:);
        n_curr = size(X_curr,1);
        Sigma_X = Sigma_X + (mean(X_curr,1)-mean(fg,1)).'*(mean(X_curr,1)-mean(fg,1))*n_curr;
    end

    Sigma_X = Sigma_X/n; % covariance of inverse curve E[X|Y]

    [coeff_SIR, D_SIR] = eig(Sigma_XX,Sigma_X);
    V_SIR = coeff_SIR(:,1:d);
    X_SIR = X*V_SIR;


    % CIR
    % also tuned number of slices for different d
    display('CIR......')
    if d<=2
        L = 10;
        partition = linspace(min(Y),max(Y),L);
        labels = zeros(n,1);
        for l = 1:n
            labels(l)=max(find((Y(l)>=partition)==1));
        end

        Lt = L;
        partitiont = linspace(min(Yt),max(Yt),L);
        labelst = zeros(m,1);
        for l= 1:m
            labelst(l)=max(find((Yt(l)>=partitiont)==1));
        end
        V_CIR = CIR(fg,labels,bg,labelst,alpha,d);
        X_CIR = X*V_CIR;
    else
        L = 4;
        partition = linspace(min(Y),max(Y),L);
        labels = zeros(n,1);
        for l = 1:n
            labels(l)=max(find((Y(l)>=partition)==1));
        end

        Lt = L;
        partitiont = linspace(min(Yt),max(Yt),L);
        labelst = zeros(m,1);
        for l= 1:m
            labelst(l)=max(find((Yt(l)>=partitiont)==1));
        end
        V_CIR = CIR(fg,labels,bg,labelst,alpha,d);
        X_CIR = X*V_CIR;
    end


    T = 10;
    for t = 1:T

        display('Training Linear Model')
        tic
        % Linear model
        mdl_raw_LM = fitlm(fg,Y);
        mdl_PCA_LM = fitlm(X_PCA,Y);
        mdl_CPCA_LM = fitlm(X_CPCA,Y);
        mdl_LDA_LM = fitlm(X_LDA,Y);
        mdl_LASSO_LM = fitlm(X_LASSO,Y);
        mdl_SIR_LM  = fitlm(X_SIR,Y);
        mdl_CIR_LM = fitlm(X_CIR,Y);
        MSE_raw_LM = mdl_raw_LM.MSE;
        MSE_PCA_LM = mdl_PCA_LM.MSE;
        MSE_CPCA_LM = mdl_CPCA_LM.MSE;
        MSE_LDA_LM = mdl_LDA_LM.MSE;
        MSE_LASSO_LM = mdl_LASSO_LM.MSE;
        MSE_SIR_LM = mdl_SIR_LM.MSE;
        MSE_CIR_LM = mdl_CIR_LM.MSE;
        MSE_LM(i,:,t) = [MSE_raw_LM,MSE_PCA_LM,MSE_CPCA_LM,MSE_LDA_LM,MSE_LASSO_LM,MSE_SIR_LM,MSE_CIR_LM];
        toc


        display('Training Regression Tree')
        tic
        % Regression Tree
        mdl_raw_Tree = fitrtree(fg,Y,'CrossVal','on');
        mdl_PCA_Tree = fitrtree(X_PCA,Y,'CrossVal','on');
        mdl_CPCA_Tree = fitrtree(X_CPCA,Y,'CrossVal','on');
        mdl_LDA_Tree = fitrtree(X_LDA,Y,'CrossVal','on');
        mdl_LASSO_Tree = fitrtree(X_LASSO,Y,'CrossVal','on');
        mdl_SIR_Tree  = fitrtree(X_SIR,Y,'CrossVal','on');
        mdl_CIR_Tree = fitrtree(X_CIR,Y,'CrossVal','on');
        MSE_raw_Tree = kfoldLoss(mdl_CIR_Tree);
        MSE_PCA_Tree = kfoldLoss(mdl_PCA_Tree);
        MSE_CPCA_Tree = kfoldLoss(mdl_CPCA_Tree);
        MSE_LDA_Tree = kfoldLoss(mdl_LDA_Tree);
        MSE_LASSO_Tree = kfoldLoss(mdl_LASSO_Tree);
        MSE_SIR_Tree = kfoldLoss(mdl_SIR_Tree);
        MSE_CIR_Tree = kfoldLoss(mdl_CIR_Tree);
        MSE_Tree(i,:,t) = [MSE_raw_Tree,MSE_PCA_Tree,MSE_CPCA_Tree,MSE_LDA_Tree,MSE_LASSO_Tree,MSE_SIR_Tree,MSE_CIR_Tree];
        toc

        display('Training SVM regression')
        tic
        % SVM
        mdl_raw_SVM = fitrsvm(fg,Y,'CrossVal','on');
        mdl_PCA_SVM = fitrsvm(X_PCA,Y,'CrossVal','on');
        mdl_CPCA_SVM = fitrsvm(X_CPCA,Y,'CrossVal','on');
        mdl_LDA_SVM = fitrsvm(X_LDA,Y,'CrossVal','on');
        mdl_LASSO_SVM = fitrsvm(X_LASSO,Y,'CrossVal','on');
        mdl_SIR_SVM  = fitrsvm(X_SIR,Y,'CrossVal','on');
        mdl_CIR_SVM = fitrsvm(X_CIR,Y,'CrossVal','on');
        MSE_raw_SVM = kfoldLoss(mdl_CIR_SVM);
        MSE_PCA_SVM = kfoldLoss(mdl_PCA_SVM);
        MSE_CPCA_SVM = kfoldLoss(mdl_CPCA_SVM);
        MSE_LDA_SVM = kfoldLoss(mdl_LDA_SVM);
        MSE_LASSO_SVM = kfoldLoss(mdl_LASSO_SVM);
        MSE_SIR_SVM = kfoldLoss(mdl_SIR_SVM);
        MSE_CIR_SVM = kfoldLoss(mdl_CIR_SVM);
        MSE_SVM(i,:,t) = [MSE_raw_SVM,MSE_PCA_SVM,MSE_CPCA_SVM,MSE_LDA_SVM,MSE_LASSO_SVM,MSE_SIR_SVM,MSE_CIR_SVM];
        toc


        display('Training Gaussian process regression')
        tic
        % Gaussian process regression
        mdl_raw_GPR = fitrgp(fg,Y,'CrossVal','on');
        mdl_PCA_GPR = fitrgp(X_PCA,Y,'CrossVal','on');
        mdl_CPCA_GPR = fitrgp(X_CPCA,Y,'CrossVal','on');
        mdl_LDA_GPR = fitrgp(X_LDA,Y,'CrossVal','on');
        mdl_LASSO_GPR = fitrgp(X_LASSO,Y,'CrossVal','on');
        mdl_SIR_GPR  = fitrgp(X_SIR,Y,'CrossVal','on');
        mdl_CIR_GPR = fitrgp(X_CIR,Y,'CrossVal','on');
        MSE_raw_GPR = kfoldLoss(mdl_CIR_GPR);
        MSE_PCA_GPR = kfoldLoss(mdl_PCA_GPR);
        MSE_CPCA_GPR = kfoldLoss(mdl_CPCA_GPR);
        MSE_LDA_GPR = kfoldLoss(mdl_LDA_GPR);
        MSE_LASSO_GPR = kfoldLoss(mdl_LASSO_GPR);
        MSE_SIR_GPR = kfoldLoss(mdl_SIR_GPR);
        MSE_CIR_GPR = kfoldLoss(mdl_CIR_GPR);
        MSE_GPR(i,:,t) = [MSE_raw_GPR,MSE_PCA_GPR,MSE_CPCA_GPR,MSE_LDA_GPR,MSE_LASSO_GPR,MSE_SIR_GPR,MSE_CIR_GPR];
        toc

        display('Training NN regression')
        tic
        % Neural Network
        mdl_raw_NN = fitrnet(fg,Y,'CrossVal','on');
        mdl_PCA_NN = fitrnet(X_PCA,Y,'CrossVal','on');
        mdl_CPCA_NN = fitrnet(X_CPCA,Y,'CrossVal','on');
        mdl_LDA_NN = fitrnet(X_LDA,Y,'CrossVal','on');
        mdl_LASSO_NN = fitrnet(X_LASSO,Y,'CrossVal','on');
        mdl_SIR_NN  = fitrnet(X_SIR,Y,'CrossVal','on');
        mdl_CIR_NN = fitrnet(X_CIR,Y,'CrossVal','on');
        MSE_raw_NN = kfoldLoss(mdl_CIR_NN);
        MSE_PCA_NN = kfoldLoss(mdl_PCA_NN);
        MSE_CPCA_NN = kfoldLoss(mdl_CPCA_NN);
        MSE_LDA_NN = kfoldLoss(mdl_LDA_NN);
        MSE_LASSO_NN = kfoldLoss(mdl_LASSO_NN);
        MSE_SIR_NN = kfoldLoss(mdl_SIR_NN);
        MSE_CIR_NN = kfoldLoss(mdl_CIR_NN);
        MSE_NN(i,:,t) = [MSE_raw_NN,MSE_PCA_NN,MSE_CPCA_NN,MSE_LDA_NN,MSE_LASSO_NN,MSE_SIR_NN,MSE_CIR_NN];
        toc
        %         display(['                     raw, ','PCA, ','CPCA, ','LDA, ','LASSO, ','SIR, ','CIR.'])
        %         display(['LM MSE: ',num2str(MSE_raw_LM),', ', num2str(MSE_PCA_LM),', ',num2str(MSE_CPCA_LM),', ',num2str(MSE_LDA_LM),', ',num2str(MSE_LASSO_LM),', ',num2str(MSE_SIR_LM),', ', num2str(MSE_CIR_LM)])
        %         display(['Decision Tree MSE: ',num2str(MSE_raw_Tree),', ', num2str(MSE_PCA_Tree),', ',num2str(MSE_CPCA_Tree),', ',num2str(MSE_LDA_Tree),', ',num2str(MSE_LASSO_Tree),', ', num2str(MSE_SIR_Tree),', ', num2str(MSE_CIR_Tree)])
        %     end
    end


end


MSE_LM_mean = mean(MSE_LM,3);

% Prediction MSE of linear regression
% other methods are slower and sub-optimal
figure
hold on
for j = 1:size(MSE_LM,2)
    if j<7
        plot(Ds,MSE_LM_mean(:,j),'--','LineWidth',4)
    else
        plot(Ds,MSE_LM_mean(:,j),'LineWidth',4)
    end
end
legend('Raw','PCA','CPCA','LDA','LASSO','SIR','CIR','FontSize',48)
hold off
title('Linear Regression MSE','FontSize',48)
set(gca, 'FontSize', 36)
xlabel('d')
ylabel('MSE')

% MSE_Tree_mean = mean(MSE_Tree,3);
% figure
% hold on
% for j = 1:size(MSE_LM,2)
%     plot(Ds,MSE_Tree_mean(:,j),'LineWidth',2)
% end
% legend('Raw','PCA','CPCA','LDA','LASSO','SIR','CIR','FontSize',48)
% hold off
% title('Tree MSE')
% 
% 
% MSE_GPR_mean = mean(MSE_GPR,3);
% figure
% hold on
% for j = 1:size(MSE_LM,2)
%     plot(Ds,MSE_GPR_mean(:,j),'LineWidth',2)
% end
% legend('Raw','PCA','CPCA','LDA','LASSO','SIR','CIR','FontSize',48)
% hold off
% title('GPR MSE')


