%一种对job1*n维聚类，一种对job矩阵聚类 
clear; clc;
load XX33.mat;load YY33.mat;
for d=1:length(X)
   XX{1,d} =X{1,d}(:,3:61);
end
for i=1:length(XX)
    A(i,1:5)=XX{1,i}(1,1:5);
    for j=6:59
        A(i,6:59)=mean(XX{1,i}(:,j));
        A(i,70:123)=std(XX{1,i}(:,j));
    end
    A(i,124)=size(XX{1,i},1);
end
A=zscore(A);
%fcm聚类
clust=4;
options=nan(4,1);
options(4)=0;
[~,U]=fcm(A,clust,options);
[~,fidx]=max(U);
c=fidx';
%计算各个类的个数
[s1,F6]=silhouette(A,c);
mean(s1)
m=[];
for j=1:max(c)
    k=0;
    for i=1:length(A)
        if c(i)== j;
            k=k+1;
        end
    end
    m(j)=k;  %各个类的个数
end