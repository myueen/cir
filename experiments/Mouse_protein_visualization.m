clear
close all


data = readtable('.../Data_Cortex_Nuclear.csv');

% foreground data
fg = data;
fg(any(ismissing(fg), 2), :) = [];
% foreground label
Y = fg.class;
Y = categorical(Y);
Y = renamecats(Y,{'c-CS-m','c-CS-s','c-SC-m','c-SC-s','t-CS-m','t-CS-s','t-SC-m','t-SC-s'},{'1','2','3','4','5','6','7','8'});
Y = str2double(string(Y));
labels = unique(Y); % set of unique foreground labels
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




% tuning parameter alpha
alpha = 0.0001;

% reduce dimension from 77 to 2 for visualization
d = 2;

% CIR
tic
display('CIR......')
V_CIR = CIR(fg,Y,bg,Yt,alpha,d);
X_CIR = X*V_CIR;
toc

cir_silhouette_values = silhouette(X_CIR, Y);
cir_mean_silhouette_score = mean(cir_silhouette_values);
fprintf('CIR Mean Silhouette Score: %.4f\n', cir_mean_silhouette_score);

cir_calinski_harabasz = CHI(X_CIR, Y);
cir_mean_calinski = mean(cir_calinski_harabasz);
fprintf('CIR Calinski Harabasz Score: %.2f\n', cir_mean_calinski);

cir_davies_bouldin = dbindex(Y, X_CIR);
cir_mean_davies = mean(cir_davies_bouldin);
fprintf('CIR Davies Bouldin Score: %.2f\n', cir_mean_davies);





display('SIR......')
tic
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
toc


sir_silhouette_values = silhouette(X_SIR, Y);
sir_mean_silhouette_score = mean(sir_silhouette_values);
fprintf('SIR Mean Silhouette Score: %.4f\n', sir_mean_silhouette_score);

sir_calinski_harabasz = CHI(X_SIR, Y);
sir_mean_calinski = mean(sir_calinski_harabasz);
fprintf('SIR Calinski Harabasz Score: %.2f\n', sir_mean_calinski);


sir_davies_bouldin = dbindex(Y, X_SIR);
sir_mean_davies = mean(sir_davies_bouldin);
fprintf('SIR Davies Bouldin Score: %.2f\n', sir_mean_davies);







% PCA
tic
display('PCA......')
[coeff,score,latent] = pca(X);
X_PCA = score(:,1:d);
toc


pca_silhouette_values = silhouette(X_PCA, Y);
pca_mean_silhouette_score = mean(pca_silhouette_values);
fprintf('PCA Mean Silhouette Score: %.4f\n', pca_mean_silhouette_score);

pca_calinski_harabasz = CHI(X_PCA, Y);
pca_mean_calinski = mean(pca_calinski_harabasz); 
fprintf('PCA Calinski Harabasz Score: %.2f\n', pca_mean_calinski);

pca_davies_bouldin = dbindex(Y, X_PCA);
pca_mean_davies = mean(pca_davies_bouldin);
fprintf('PCA Davies Bouldin Score: %.2f\n', pca_mean_davies);







% CPCA
tic
display('CPCA......')
alpha_CPCA = 2; %
[coeff_CPCA,D_CPCA] = eig(cov(fg)-alpha_CPCA*cov(bg));
V_CPCA = coeff_CPCA(:,1:d);
X_CPCA = X*V_CPCA;
toc

cpca_silhouette_values = silhouette(X_CPCA, Y);
cpca_mean_silhouette_score = mean(cpca_silhouette_values);
fprintf('CPCA Mean Silhouette Score: %.4f\n', cpca_mean_silhouette_score);

cpca_calinski_harabasz = CHI(X_CPCA, Y);
cpca_mean_calinski = mean(cpca_calinski_harabasz);
fprintf('CPCA Calinski Harabasz Score: %.2f\n', cpca_mean_calinski);

cpca_davies_bouldin = dbindex(Y, X_CPCA);
cpca_mean_davies = mean(cpca_davies_bouldin);
fprintf('CPCA Davies Bouldin Score: %.2f\n', cpca_mean_davies);





% tSNE
tic
X_tSNE = tsne(fg);
toc

tSNE_silhouette_values = silhouette(X_tSNE, Y);
tSNE_mean_silhouette_score = mean(tSNE_silhouette_values);
fprintf('tSNE Mean Silhouette Score: %.4f\n', tSNE_mean_silhouette_score);

tSNE_calinski_harabasz = CHI(X_tSNE, Y);
tSNE_mean_calinski = mean(tSNE_calinski_harabasz);
fprintf('tSNE Calinski Harabasz Score: %.2f\n', tSNE_mean_calinski);

tSNE_davies_bouldin = dbindex(Y, X_tSNE);
tSNE_mean_davies = mean(tSNE_davies_bouldin);
fprintf('tSNE Davies Bouldin Score: %.2f\n', tSNE_mean_davies);




% UMAP
tic
X_UMAP = run_umap(fg);
toc

UMAP_silhouette_values = silhouette(X_UMAP, Y);
UMAP_mean_silhouette_score = mean(UMAP_silhouette_values);
fprintf('UMAP Mean Silhouette Score: %.4f\n', UMAP_mean_silhouette_score);

umap_calinski_harabasz = CHI(X_UMAP, Y);
umap_mean_calinski = mean(umap_calinski_harabasz);
fprintf('UMAP Calinski Harabasz Score: %.2f\n', umap_mean_calinski);

umap_davies_bouldin = dbindex(Y, X_UMAP);
umap_mean_davies = mean(umap_davies_bouldin);
fprintf('UMAP Davies Bouldin Score: %.2f\n', umap_mean_davies);




% % supervised UMAP
% X_SUMAP = run_umap([Y,fg],'label_column',1);



% LDA
tic
[X_LDA,V_LDA,lambda_LDA] = LDA(fg,Y);
X_LDA = fg*V_LDA(:,1:d);
toc

LDA_silhouette_values = silhouette(X_LDA, Y);
LDA_mean_silhouette_score = mean(LDA_silhouette_values);
fprintf('LDA Mean Silhouette Score: %.4f\n', LDA_mean_silhouette_score);

lda_calinski_harabasz = CHI(X_LDA, Y);
lda_mean_calinski = mean(lda_calinski_harabasz);
fprintf('LDA Calinski Harabasz Score: %.2f\n', lda_mean_calinski);

lda_davies_bouldin = dbindex(Y, X_LDA);
lda_mean_davies = mean(lda_davies_bouldin);
fprintf('LDA Davies Bouldin Score: %.2f\n', lda_mean_davies);



% LASSO
tic
B = lasso(X,Y);
B_sum = sum(B~=0,1);
lambda_idx = min(find(B_sum<=d));
selected_idx = find(B(:,lambda_idx)~=0);
X_LASSO = X(:,selected_idx);
toc


LASSO_silhouette_values = silhouette(X_LASSO, Y);
LASSO_mean_silhouette_score = mean(LASSO_silhouette_values);
fprintf('LASSO Mean Silhouette Score: %.4f\n', LASSO_mean_silhouette_score);

lasso_calinski_harabasz = CHI(X_LASSO, Y);
lasso_mean_calinski = mean(lasso_calinski_harabasz);
fprintf('LASSO Calinski Harabasz Score: %.2f\n', lasso_mean_calinski);


lasso_davies_bouldin = dbindex(Y, X_LASSO);
lasso_mean_davies = mean(lasso_davies_bouldin);
fprintf('LASSO Davies Bouldin Score: %.2f\n', lasso_mean_davies);





% colors credit to Color Brewer
colors = [228,26,28;55,126,184;77,175,74;152,78,163;255,127,0;255,255,51;166,86,40;247,129,191]/255;
markers = {'o','^','square','pentagram','o','^','square','pentagram'};

figure
subplot(2,4,1)
hold on
for l = 1:L
    X_curr = X_PCA(find(Y==labels(l)),:);
    scatter(X_curr(:,1),X_curr(:,2),10,'filled','MarkerFaceColor',colors(l,:),'Marker',markers{l})
end
[~, objh] = legend({'c-CS-m','c-CS-s','c-SC-m','c-SC-s','t-CS-m','t-CS-s','t-SC-m','t-SC-s'}, 'location', 'West', 'Fontsize', 14);
objhl = findobj(objh, 'type', 'line'); %// objects of legend of type line
set(objhl, 'Markersize', 14); %// set marker size as desired
% or for Patch plots 
objhl = findobj(objh, 'type', 'patch'); % objects of legend of type patch
set(objhl, 'Markersize', 14); % set marker size as desired


hold off
title('PCA','FontSize', 16)
set(gca, 'FontSize', 16)

subplot(2,4,2)
hold on
for l = 1:L
    X_curr = X_CPCA(find(Y==labels(l)),:);
    scatter(X_curr(:,1),X_curr(:,2),10,'filled','MarkerFaceColor',colors(l,:),'Marker',markers{l})
end
hold off
title('CPCA','FontSize', 16)
set(gca, 'FontSize', 16)


subplot(2,4,3)
hold on
for l = 1:L
    X_curr = X_LDA(find(Y==labels(l)),:);
    scatter(X_curr(:,1),X_curr(:,2),10,'filled','MarkerFaceColor',colors(l,:),'Marker',markers{l})
end
hold off
title('LDA','FontSize', 16)
set(gca, 'FontSize', 16)

subplot(2,4,4)
hold on
for l = 1:L
    X_curr = X_LASSO(find(Y==labels(l)),:);
    scatter(X_curr(:,1),X_curr(:,2),10,'filled','MarkerFaceColor',colors(l,:),'Marker',markers{l})
end
hold off
title('LASSO','FontSize', 16)
set(gca, 'FontSize', 16) 


subplot(2,4,5)
hold on
for l = 1:L
    X_curr = X_SIR(find(Y==labels(l)),:);
    scatter(X_curr(:,1),X_curr(:,2),10,'filled','MarkerFaceColor',colors(l,:),'Marker',markers{l})
end
hold off
title('SIR','FontSize', 16)
set(gca, 'FontSize', 16)

subplot(2,4,6)
hold on
for l = 1:L
    X_curr = X_CIR(find(Y==labels(l)),:);
    scatter(X_curr(:,1),X_curr(:,2),10,'filled','MarkerFaceColor',colors(l,:),'Marker',markers{l})
end
hold off
title('CIR','FontSize', 16)   % default(initial - 32
set(gca, 'FontSize', 16)      % 32


subplot(2,4,7)
hold on
for l = 1:L
    X_curr = X_tSNE(find(Y==labels(l)),:);
    scatter(X_curr(:,1),X_curr(:,2),10,'filled','MarkerFaceColor',colors(l,:),'Marker',markers{l})
end
hold off
title('tSNE','FontSize', 16)
set(gca, 'FontSize', 16)


subplot(2,4,8)
hold on
for l = 1:L
    X_curr = X_UMAP(find(Y==labels(l)),:);
    scatter(X_curr(:,1),X_curr(:,2),10,'filled','MarkerFaceColor',colors(l,:),'Marker',markers{l})
end
hold off
title('UMAP','FontSize', 16)
set(gca, 'FontSize', 16) % original 32

% subplot(3,3,9)
% hold on
% for l = 1:L
%     X_curr = X_SUMAP(find(Y==labels(l)),:);
%     scatter(X_curr(:,1),X_curr(:,2),100,'filled','MarkerFaceColor',colors(l,:),'Marker',markers{l})
% end
% hold off
% title('Supervised UMAP','FontSize', 32)
% set(gca, 'FontSize', 32)


% some measures of clustering, although biased
display('Silhouette scores for raw, PCA, CPCA, LDA, LASSO, SIR, CIR, tSNE, UMAP')
display(['                     ',num2str(mean(silhouette(X,Y))), ', ',num2str(mean(silhouette(X_PCA,Y))), ', ',num2str(mean(silhouette(X_CPCA,Y))), ', ',num2str(mean(silhouette(X_LDA,Y))), ', ',num2str(mean(silhouette(X_LASSO,Y))), ', ',num2str(mean(silhouette(X_SIR,Y))), ', ',num2str(mean(silhouette(X_CIR,Y))), ', ',num2str(mean(silhouette(X_tSNE,Y))), ', ',num2str(mean(silhouette(X_UMAP,Y)))])
display('Calinski-Harabasz Index for raw, PCA, CPCA, LDA, LASSO, SIR, CIR, tSNE, UMAP,SUMAP')
display(['                     ',num2str(mean(CHI(X,Y))), ', ',num2str(mean(CHI(X_PCA,Y))), ', ',num2str(mean(CHI(X_CPCA,Y))), ', ',num2str(mean(CHI(X_LDA,Y))), ', ',num2str(mean(CHI(X_LASSO,Y))), ', ',num2str(mean(CHI(X_SIR,Y))), ', ',num2str(mean(CHI(X_CIR,Y))), ', ',num2str(mean(CHI(X_tSNE,Y))), ', ',num2str(mean(CHI(X_UMAP,Y)))])

display('Davies Bouldin Index for raw, PCA, CPCA, LDA, LASSO, SIR, CIR, tSNE, UMAP')
display(['',num2str(mean(dbindex(Y,X))), ', ',num2str(mean(dbindex(Y, X_PCA))), ', ',num2str(mean(dbindex(Y,X_CPCA))), ', ',num2str(mean(dbindex(Y,X_LDA))), ', ',num2str(mean(dbindex(Y,X_LASSO))), ', ',num2str(mean(dbindex(Y,X_SIR))), ', ',num2str(mean(dbindex(Y,X_CIR))), ', ',num2str(mean(dbindex(Y,X_tSNE)))])