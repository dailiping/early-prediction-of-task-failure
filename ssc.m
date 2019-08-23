% This is the main function for running SSC.
load XX33.mat;load YY33.mat;
for d=1:length(X)
   XX{1,d} =X{1,d}(:,3:61);
end
data=XX{1,1};
%data=importdata('1.mat')
%clc, clear all, close all
D = 100; %Dimension of ambient space环境空间的维度
n = 2;%Number of subspaces数量的子空间
d1 = 1; d2 = 1;%d1 and d2: dimension of subspace 1 and 2维子空间1和2
N1 = 100; N2 = 100;%N1 and N2: number of points in subspace 1 and 2子空间的点1和2
X1 = randn(D,d1) * randn(d1,N1);%Generating N1 points in a d1 dim. subspace 将N1点生成在D1的子空间
X2 = randn(D,d2) * randn(d2,N2);%Generating N2 points in a d2 dim. subspace 将N2点生成在D2的子空间
X = [X1 X2];
s = [1*ones(1,N1) 2*ones(1,N2)]; %Generating the ground-truth for evaluating clustering results生成真实评价聚类结果
r =200;%Enter the projection dimension e.g. r = d*n, enter r = 0 to not project输入投影尺寸r=d*n，r=0是不进行投影
Cst = 1;%Enter 1 to use the additional affine constraint sum(c) == 1 输入1使用额外的放射约束
OptM = 'Lasso';%OptM can be {'L1Perfect','L1Noise','Lasso','L1ED'} 有四种形式
lambda = 0.001;%Regularization parameter in 'Lasso' or the noise level for 'L1Noise' 正则化参数或积分水平
K = max(d1,d2);%Number of top coefficients to build the similarity graph, enter K=0 for using the whole coefficients K=0使用整个系数
if Cst == 1
    K = max(d1,d2) + 1;%For affine subspaces, the number of coefficients to pick is dimension + 1 放射子空间维数是1
end

Xp = DataProjection(X,r,'NormalProj');
CMat = SparseCoefRecovery(Xp,Cst,OptM,lambda);
[CMatC,sc,OutlierIndx,Fail] = OutlierDetection(CMat,s);
if (Fail == 0)
    CKSym = BuildAdjacency(CMatC,K);
    [Grps , SingVals, LapKernel] = SpectralClustering(CKSym,n);
    Missrate = Misclassification(Grps,sc);
    save Lasso_001.mat CMat CKSym Missrate SingVals LapKernel Fail
else
    save Lasso_001.mat CMat Fail
end

