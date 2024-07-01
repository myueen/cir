function vrc = CHI(X, Y)
% Calculate Calinski-Harabasz Criterion for evaluating the optimal k for a
% given clusteing solution. Works best with kmean clustering and squared euclidean distance.
% 
% See Calinski, R. B., and Harabasz, J. (1974) A Dendrite Method for Cluster Analysis, Communications in Statistics, 3, 1-27.
%
% vrc = CalinskiHarabasz(X, IDX, C, SUMD)
%
% Inputs:
% ---------------------------------------------------------------------
% X                         : Matrix used for clustering 
%
% Y                       : Cluster Labels from clustering output

%
% Outputs:
% ---------------------------------------------------------------------
% vrc                       : Validity criterion
%


%Number of Clusters
if min(Y)>0
    Y = Y-ones(size(Y));
end
clusts = unique(Y);
N = length(clusts);
C = zeros(N,size(X,2));
n = size(X,1);
Ni= accumarray(Y+ones(size(Y)),ones(length(Y),1));

for i = 1:N
    X_curr = X(find(Y==clusts(i)),:);
    C(i,:) = mean(X_curr,1);
end


SUMD = (pdist2(X,C)).^2;

%Calculate within sum of squares
%SUMD is the sum of squared Euclidean
%Distance
ssw = 0;
for i =1:n
    %ssw = ssw + SUMD(i,Y(i)+1);
    ssw = ssw + (X(i,:)-C(Y(i)+1,:))*(X(i,:)-C(Y(i)+1,:)).';
end

%Calculate Between sum of squares
ssb = sum(Ni.*(pdist2(C,mean(X))).^2);

vrc = (ssb/ssw) * (n-N)/(N-1);