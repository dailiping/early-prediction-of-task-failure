%{
%% file example_Robust.m
% this file shows the usage of Least_RMTL.m function 
% and study how to detect outlier tasks. 
%
%% OBJECTIVE
% argmin_W { sum_i^t (0.5 * norm (Y{i} - X{i}' * W(:, i))^2)
%            + rho1 * \|L\|_* + rho2 * \|S\|_{1, 2}}
% where W = L + S
%       \|S\|_{1, 2} = sum( sum(S.^2) .^ 0.5 )
%       \|L\|_*      = sum( svd(L, 0) )
%
%% LICENSE
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You should have received a copy of the GNU General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%   Copyright (C) 2011 - 2012 Jiayu Zhou and Jieping Ye 
%
%   You are suggested to first read the Manual.
%   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
%   Last modified on June 3, 2012.
%
%% Related papers
%
% [1] Chen, J., Zhou, J. and Ye, J. Integrating Low-Rank and Group-Sparse
% Structures for Robust Multi-Task Learning, KDD 2011
%}
clear;clc;
%load XX11.mat;load YY11.mat;
%load XX22.mat;load YY22.mat;
load XX33.mat;load YY33.mat;
%带资源的数据
%load Xziyuan33.mat;load YY33.mat;
%load Xziyuan22.mat;load YY22.mat;
%load Xziyuan11.mat;load YY11.mat;
%三种资源合并在一起
%load XXziyuanhebing33.mat;load YY33.mat;
%62-67维度62、64、66；63、65、67
u=0;r=0;
for i=1:length(Y)
    for j=1:size(Y{1,i})
        if Y{1,i}(j,1) == 4
            Y{1,i}(j,1)=1;
            %{
             Y{1,i}(j,2)=X{1,i}(j,62);
             Y{1,i}(j,3)=X{1,i}(j,64);
             Y{1,i}(j,4)=X{1,i}(j,66);
             %}
           % Y{1,i}(j,4)=X{1,i}(j,14);
           %  Y{1,i}(j,5)=X{1,i}(j,15);
            u=u+1;
        else
             Y{1,i}(j,1)=-1;
             %{
             Y{1,i}(j,2)=X{1,i}(j,63);
             Y{1,i}(j,3)=X{1,i}(j,65);
             Y{1,i}(j,4)=X{1,i}(j,67);
             %}
            %  Y{1,i}(j,4)=X{1,i}(j,14);
            %  Y{1,i}(j,5)=X{1,i}(j,15);
             r=r+1;
        end
    end
end
for p=1:length(X)
   XX{1,p} =X{1,p}(:,3:61);
    XX{1,p} = zscore(XX{1,p});  % normalization归一化
end
%分类
%
%load('cc4.mat')
%load cengci-c.mat  %c=a;
%5类
%load fcm-c5.mat
%4类
load fcm-c.mat
%3类
%load fcm-c3.mat
%加上奇异值之后聚成4类
%load fcm-svd4
%load fcm-csvd5
%load kmean-c.mat
%load cengci-c.mat
%各个类的个数
for j=min(c):max(c)
    k=0;
    for i=1:length(XX)
        if c(i)== j;
            k=k+1;
        end
    end
    m(j)=k;  %各个类的个数
end
 k=1;%s=1;
for i=1:length(XX)
    if c(i)==1;  %更改c，c为类别数
        %XA1{k}(:,6:59)=XX{i}(:,6:59);
        XA1{k}=XX{i};
        YA1{k}=Y{i};
        k=k+1;
    end
end
%}
%计算含有多少task
%{
for i=1:length(XX)
    row(i)=size(XX{1,i},1);
end
all=sum(row);
amin=min(row);
abig=max(row);
%}
%原参数
%{
dimension = 500;
sample_size = 50;
task = 50;
X = cell(task ,1);
Y = cell(task ,1);
for i = 1: task
    X{i} = rand(sample_size, dimension);
    Y{i} = rand(sample_size, 1);
end
%}
opts = []; opts.init = 0;      % guess start point from data. 
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-6;   % tolerance. 
opts.maxIter = 2000; % maximum iteration number of optimization.
cv_fold=3;
%   rho1: low rank component L trace-norm regularization parameter
%   rho2: sparse component S L1,2-norm sprasity controlling parameter
training_percent = 0.7;
%[X_tr, Y_tr, X_te, Y_te] = mtSplitPerc(XX, Y, training_percent);
[X_tr, Y_tr, X_te, Y_te] = mtSplitPerc(XA1, YA1, training_percent);
eval_func_str = 'eval_MTL_acc';higher_better = true;
%param_range = [0.1 1 10 30 50 70 90 110];
%param_range = [0.001 0.01 0.1 1 10 100 1000 10000]; 
%param_range = [50 100 150 200 250 300 350 400 450 500 600 700]; %200 20
%{
param_range1 = [50 100 150 200 250 300 350 400 450 500 600];
param_range2 = [10 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 180 200];
%}
param_range = [10 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 180 200]; 
%param_range = [0.01 0.01 0.1 10 20 30 40 50 100 150 200 250 300 350 400 450 500 600 700 1000 10000];
%rho_2=70;rho_1=10;
fprintf('Perform model selection via cross validation: \n')
%[best_param1,best_param2,AA] = CrossValidationParam1( X_tr, Y_tr, 'Least_RMTL',param_range1, param_range2,opts, cv_fold, eval_func_str, higher_better)
%调第一个参数
%{
rho_2 = 10;  %2=90 1=60 %2=70 1=60
best_param = CrossValidationParam1( X_tr, Y_tr, 'Least_RMTL' ,param_range,rho_2,opts, cv_fold, eval_func_str, higher_better)
test_time=tic;
[W,funcVal,L,S] = Least_RMTL(X_tr, Y_tr,best_param,rho_2,opts);
testTime=toc(test_time);
%115  第四类
%}
%
%调第二个参数
%
rho_1 = 50; %115 90
best_param = CrossValidationParam1( X_tr, Y_tr, 'Least_RMTL', param_range,rho_1,opts, cv_fold, eval_func_str, higher_better)
test_time=tic;
[W,funcVal,L,S] = Least_RMTL(X_tr, Y_tr,rho_1, best_param,opts);
testTime=toc(test_time);
%}
%交叉调参
%{
[best_param1,best_param2] = CrossValidationParam2( X_tr, Y_tr, 'Least_RMTL', param_range1,param_range2,opts, cv_fold, eval_func_str, higher_better)
test_time=tic;
[W,funcVal,L,S] = Least_RMTL(X_tr, Y_tr,best_param1, best_param2,opts);
testTime=toc(test_time);
%150,80 第四类第一次调参
%}
%循环调参
%{
[best_param1,best_param2,perform_mat1] = CrossValidationParam3(X_tr,Y_tr,'Least_RMTL',param_range1,param_range2,opts, cv_fold, eval_func_str, higher_better)
test_time=tic;
[W,funcVal,L,S] = Least_RMTL(X_tr, Y_tr,best_param1, best_param2,opts);
testTime=toc(test_time);
%z是一个矩阵,精度三维图
[e1,e2]=meshgrid(param_range1,param_range2);
surf(e1,e2,perform_mat1);
xlabel('α参数');
ylabel('β参数');
zlabel('精度');
zmax=max(max(perform_mat1));   %找出Z的最大值zmax
%[id_ymax,id_xmax]=find(Z==zmax);
xmax=best_param1;
ymax=best_param2;   %找出Z的最大值对应的横纵坐标xmax、ymax
hold on
plot3(xmax,ymax,zmax,'k.','markersize',20)   %标记一个黑色的圆点
text(xmax,ymax,zmax,['  x=',num2str(xmax),char(10),'  y=',num2str(ymax),char(10),'  z=',num2str(zmax)]);   %标出坐标
%}
%axis([0 800 0 1000 0.80 0.1]);
%求资源的占比
%{
for i=1:length(Y_tr)
    for j=1:size(Y_tr{1,i},1)
        Y_tr1{1,i}(j,1)=Y_tr{1,i}(j,1);
    end
end
%}
%不调参
%{
rho_1=10;rho_2=70;
test_time=tic;
[W,funcVal,L,S] = Least_RMTL(X_tr, Y_tr, rho_1, rho_2, opts);
testTime=toc(test_time);
%}

%[W funcVal L S] = Least_RMTL(X_tr, Y_tr, best_param1, best_param2,  opts);
%sum(sum(S(1,:)~=0))

%{
for i=1:length(Y_te)
    for j=1:size(Y_te{1,i},1)
    Y_te1{1,i}(j,1)=Y_te{1,i}(j,1);
    end
end
%}
[final_performance,predLabel,origLable] = eval_MTL_acc(Y_te, X_te, W);%得到精确度
%计算相对节省的资源
%之前的算法
%{
s=0;t=0;p=0;
for i=1:length(predLabel)
    %
    for j=1:size(predLabel{1,i},1)
        p=p+1;
       if  predLabel{1,i}(j,1)== -1 && Y_te{1,i}(j,1)== 1
           s=s+1;
           %jian(s,1)=Y_te{1,i}(j,2);
           %jian1(s,1)=Y_te{1,i}(j,3);
          % jian2(s,1)=Y_te{1,i}(j,4);
          % jian3(s,1)=Y_te{1,i}(j,5);
          minus(s,1)=Y_te{1,i}(j,2); %浪费的资源
          minus1(s,1)=Y_te{1,i}(j,3);
       end
       if  predLabel{1,i}(j,1)== -1 && Y_te{1,i}(j,1)== -1
           t=t+1;
           %jia(t,1)=Y_te{1,i}(j,2);
           %jia1(t,1)=Y_te{1,i}(j,3);
          % jia2(t,1)=Y_te{1,i}(j,4);
          % jia3(t,1)=Y_te{1,i}(j,5);
          add(t,1)=Y_te{1,i}(j,3)-Y_te{1,i}(j,2);%-Y_te{1,i}(j,2);%节省的资源
          add1(t,1)=Y_te{1,i}(j,4)-Y_te{1,i}(j,2);%-Y_te{1,i}(j,2);
       end
       all(p,1)=Y_te{1,p}(:,4);
       JA=add/all;JA1=add1/all;
       JAN=minus/all;JAN1=minus1/all;
    end
    A(1,i)=SUM(JA)-SUM(JAN);
    A1(1,i)=SUM(JA1)-SUM(JAN1);
    %all(i,1)=sum(Y_te{1,i}(1:size(predLabel{1,i},1),2));
    %all1(i,1)=sum(Y_te{1,i}(1:size(predLabel{1,i},1),3));
   % all2(i,1)=sum(Y_te{1,i}(1:size(predLabel{1,i},1),4));
   % all3(i,1)=sum(Y_te{1,i}(1:size(predLabel{1,i},1),5));
end
AA=sum(A);
AA1=sum(A1);

% Rjia=sum(jia);Rjian=sum(jian);Rall=sum(all);
% AA0=(Rjia-Rjian)/Rall
% Rjia1=sum(jia1);Rjian1=sum(jian1);Rall1=sum(all1);
% AA1=(Rjia1-Rjian1)/Rall1
% Rjia2=sum(jia2);Rjian2=sum(jian2);Rall2=sum(all2);
% AA2=(Rjia2-Rjian2)/Rall2
% Rjia3=sum(jia3);Rjian3=sum(jian3);Rall3=sum(all3);
% AA3=(Rjia3-Rjian3)/Rall3
%}
%现在的算法
%Y里面有四列值，第一列：真实值，第二列：1/3，第三列：1/2，第四列：1
%{
s=0;t=0;p=0;x=0;x1=0;
for i=1:length(predLabel)
    JA3=[];JA2=[];J3=[];J2=[];
    for j=1:size(predLabel{1,i},1)
        p=p+1;
        minus3=[];minus2=[];all=[];add3=[];add2=[];All=[];
        if  predLabel{1,i}(j,1)== -1 && Y_te{1,i}(j,1)== 1
            s=s+1;%有956
            minus3(s,1)=Y_te{1,i}(j,2); %浪费的资源
            minus2(s,1)=Y_te{1,i}(j,3);
            all(s,1)=Y_te{1,i}(j,4);
            AW3(s,1)=Y_te{1,i}(j,2);
            AW2(s,1)=Y_te{1,i}(j,3);
            if all(s,1)==0 %有两个x1=0；
                x1=x1+1;
                J3(s,1)=0;J2(s,1)=0;
            else
                J3(s,1)=minus3(s,1)/all(s,1);
                J2(s,1)=minus2(s,1)/all(s,1);
            end            
        end
        if  predLabel{1,i}(j,1)== -1 && Y_te{1,i}(j,1)== -1
            t=t+1;%有7507
            add3(t,1)=Y_te{1,i}(j,4)-Y_te{1,i}(j,2);%节省的资源
            add2(t,1)=Y_te{1,i}(j,4)-Y_te{1,i}(j,3);
            All(t,1)=Y_te{1,i}(j,4);
            AB3(t,1)=Y_te{1,i}(j,2);
            AB2(t,1)=Y_te{1,i}(j,3);
            if All(t,1)==0 %2584个x等于0
                x=x+1;%在最后一个时段为0，前1/3、1/2时段的值也为0
                JA3(t,1)=0;JA2(t,1)=0;
            else
                JA3(t,1)=add3(t,1)/All(t,1);
                JA2(t,1)=add2(t,1)/All(t,1);
            end
            
        end
    end
    %一个task的占比
    B(1,i)=sum(JA3);C(1,i)=sum(J3);
    D(1,i)=sum(JA2);E(1,i)=sum(J2);
    %所有资源和
    B1(1,i)=sum(add3);C1(1,i)=sum(minus3);
    D1(1,i)=sum(add2);E1(1,i)=sum(minus2);
    %AS(1,i)=sum(All);AM(1,i)=sum(all);
    AS(1,i)=sum(AW3)+sum(AB3);AM(1,i)=sum(AW2)+sum(AB2);
end
%一个task的占比
AA=sum(B)/t;AA1=sum(C)/s;
AA2=sum(D)/t;AA3=sum(E)/s;
%所有资源的占比
ZI3=sum(B1)/sum(AS);ZI2=sum(D1)/sum(AM);
%ZI2=sum(C1)/sum(AS);ZI2=sum(DE1)/sum(AM);
%ADD=AA-AA1;
%MINUS=AA2-AA3;
%}
sumj=1;
sumk=1;
for i=1:length(origLable)
    predict_labelElmAe=origLable{1,i};
    row=length(predict_labelElmAe);
    for j=1:row
       if predict_labelElmAe(j,1)>0
           actual(1,sumj)=4;
       else
           actual(1,sumj)=5;
       end
       sumj=sumj+1;
    end
    test_labelElmAe=predLabel{1,i};
    row=length(test_labelElmAe);
    for k=1:row
       if test_labelElmAe(k,1)>0
           expect(1,sumk)=4;
       else
           expect(1,sumk)=5;
       end
       sumk=sumk+1;
    end
end
[Precision,FNR,Recall,Accuracy,specificity,sensitivity,F1,FPR,TPR]=Evolution_result(expect,actual);
AQQ=[Accuracy F1 FNR FPR testTime Recall];
%}
% draw figure
%{
close;
figure();
subplot(3,1,1);
%imshow(1- (abs(S')~=0), 'InitialMagnification', 'fit');
imshow((abs(S')/max(max(abs(S')))), 'InitialMagnification', 'fit')
ylabel('S^T (outliers)');
title('Visualization of Robust Multi-Task Learning Model');
subplot(3,1,2);
%imshow(1- (zscore(L')), 'InitialMagnification', 'fit')
imshow((abs(L')/max(max(abs(L')))), 'InitialMagnification', 'fit')
ylabel('L^T (low rank)');
subplot(3,1,3);
%imshow(1- (zscore(W')), 'InitialMagnification', 'fit')
imshow((abs(W')/max(max(abs(W')))), 'InitialMagnification', 'fit')
ylabel('W^T');
xlabel('Dimension')
colormap(jet) %Jet
print('-dpdf', '-r600', 'LeastRMTLExp');
%}