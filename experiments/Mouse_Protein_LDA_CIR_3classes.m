clear
close all


data = readtable('.../Data_Cortex_Nuclear.csv');

% foreground data
fg = data;
fg(any(ismissing(fg), 2), :) = [];
% foreground label
Y = fg.class;

% extract samples for classes
classes_cs = {'c-CS-m', 'c-CS-s', 't-CS-m'};
is_Y_cs = ismember(Y, classes_cs);
Y_3classes = Y(is_Y_cs,:);
Y_3classes = categorical(Y_3classes);
Y_3classes = renamecats(Y_3classes, {'c-CS-m', 'c-CS-s', 't-CS-m'},{'1','2','3'});
Y_3classes = str2double(string(Y_3classes));
labels_3 = unique(Y_3classes);



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

% Set the seed for reproducibility
seed = 42;
rng(seed);



% CIR
tic
display('CIR......')
V_CIR = CIR(fg,Y,bg,Yt,alpha,d);
X_CIR = X*V_CIR;
toc

X_CIR_3classes = X_CIR(is_Y_cs, :);

cir_3c_silhouette_values = silhouette(X_CIR_3classes, Y_3classes);
cir_3c_mean_silhouette_score = mean(cir_3c_silhouette_values);
fprintf('CIR 3classes Mean Silhouette Score: %.4f\n', cir_3c_mean_silhouette_score);

cir_3c_calinski_harabasz = CHI(X_CIR_3classes, Y_3classes);
cir_3c_mean_calinski = mean(cir_3c_calinski_harabasz);
fprintf('CIR 3classes Calinski Harabasz Score: %.2f\n', cir_3c_mean_calinski);

cir_3c_davies_bouldin = dbindex(Y_3classes, X_CIR_3classes);
cir_3c_mean_davies = mean(cir_3c_davies_bouldin);
fprintf('CIR 3classes Davies Bouldin Score: %.2f\n', cir_3c_mean_davies);



% LDA
tic
[X_LDA,V_LDA,lambda_LDA] = LDA(fg,Y);
X_LDA = fg*V_LDA(:,1:d);
toc

X_LDA_3classes = X_LDA(is_Y_cs, :);


LDA_3c_silhouette_values = silhouette(X_LDA_3classes, Y_3classes);
LDA_3c_mean_silhouette_score = mean(LDA_3c_silhouette_values);
fprintf('LDA 3classes Mean Silhouette Score: %.4f\n', LDA_3c_mean_silhouette_score);

lda_3c_calinski_harabasz = CHI(X_LDA_3classes, Y_3classes);
lda_3c_mean_calinski = mean(lda_3c_calinski_harabasz);
fprintf('LDA 3classes Calinski Harabasz Score: %.2f\n', lda_3c_mean_calinski);

lda_3c_davies_bouldin = dbindex(Y_3classes, X_LDA_3classes);
lda_3c_mean_davies = mean(lda_3c_davies_bouldin);
fprintf('LDA 3classes Davies Bouldin Score: %.2f\n', lda_3c_mean_davies);



% colors credit to Color Brewer
colors = [228,26,28;55,126,184;77,175,74]/255;
markers = {'o','^','square'};



% LDA
subplot(1,2,1)
hold on
for l = 1:3
    X_curr = X_LDA_3classes(find(Y_3classes==labels_3(l)),:);
    scatter(X_curr(:,1),X_curr(:,2),10,'filled','MarkerFaceColor',colors(l,:),'Marker',markers{l})
end
hold off
title('LDA','FontSize', 16)
set(gca, 'FontSize', 16)


% CIR
subplot(1,2,2)
hold on
for l = 1:3
    X_curr = X_CIR_3classes(find(Y_3classes==labels_3(l)),:);
    scatter(X_curr(:,1),X_curr(:,2),10,'filled','MarkerFaceColor',colors(l,:),'Marker',markers{l})
end
hold off
title('CIR','FontSize', 16)   
set(gca, 'FontSize', 16)      




% some measures of clustering, although biased
display('Silhouette scores for LDA, CIR')
display([num2str(mean(silhouette(X_LDA_3classes,Y_3classes))),', ',num2str(mean(silhouette(X_CIR_3classes,Y_3classes)))])
display('Calinski-Harabasz Index for LDA, CIR')
display([num2str(mean(CHI(X_LDA_3classes,Y_3classes))), ', ',num2str(mean(CHI(X_CIR_3classes,Y_3classes)))])
display('Davies Bouldin Index for LDA, CIR')
display([num2str(mean(dbindex(Y_3classes,X_LDA_3classes))), ', ',num2str(mean(dbindex(Y_3classes,X_CIR_3classes)))])