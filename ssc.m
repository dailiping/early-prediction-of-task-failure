% This is the main function for running SSC.
load XX33.mat;load YY33.mat;
for d=1:length(X)
   XX{1,d} =X{1,d}(:,3:61);
end
data=XX{1,1};
%data=importdata('1.mat')
%clc, clear all, close all
D = 100; %Dimension of ambient space�����ռ��ά��
n = 2;%Number of subspaces�������ӿռ�
d1 = 1; d2 = 1;%d1 and d2: dimension of subspace 1 and 2ά�ӿռ�1��2
N1 = 100; N2 = 100;%N1 and N2: number of points in subspace 1 and 2�ӿռ�ĵ�1��2
X1 = randn(D,d1) * randn(d1,N1);%Generating N1 points in a d1 dim. subspace ��N1��������D1���ӿռ�
X2 = randn(D,d2) * randn(d2,N2);%Generating N2 points in a d2 dim. subspace ��N2��������D2���ӿռ�
X = [X1 X2];
s = [1*ones(1,N1) 2*ones(1,N2)]; %Generating the ground-truth for evaluating clustering results������ʵ���۾�����
r =200;%Enter the projection dimension e.g. r = d*n, enter r = 0 to not project����ͶӰ�ߴ�r=d*n��r=0�ǲ�����ͶӰ
Cst = 1;%Enter 1 to use the additional affine constraint sum(c) == 1 ����1ʹ�ö���ķ���Լ��
OptM = 'Lasso';%OptM can be {'L1Perfect','L1Noise','Lasso','L1ED'} ��������ʽ
lambda = 0.001;%Regularization parameter in 'Lasso' or the noise level for 'L1Noise' ���򻯲��������ˮƽ
K = max(d1,d2);%Number of top coefficients to build the similarity graph, enter K=0 for using the whole coefficients K=0ʹ������ϵ��
if Cst == 1
    K = max(d1,d2) + 1;%For affine subspaces, the number of coefficients to pick is dimension + 1 �����ӿռ�ά����1
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

