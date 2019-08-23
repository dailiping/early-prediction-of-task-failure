function [best_param1,best_param2,perform_mat1] = CrossValidationParam3...
    ( X, Y, obj_func_str,param_range1,param_range2, obj_func_opts, cv_fold, eval_func_str, higher_better)
%循环调参，画出精度的三维图
eval_func = str2func(eval_func_str);%将函数调换使用 eval_MTL_mse1
obj_func  = str2func(obj_func_str);  %L_T
% compute sample size for each task
task_num = length(X);
% performance vector
%perform_mat = zeros(length(param_range2),1);
% begin cross validation
fprintf('[')
perform_mat1=[];
for i=1:length(param_range1)
    rho=param_range1(i);
    perform_mat = zeros(length(param_range2),1);
for cv_idx = 1: cv_fold
    
    fprintf('.')
    % buid cross validation data splittings for each task.
    %
    cv_Xtr = cell(task_num, 1);
    cv_Ytr = cell(task_num, 1);
    cv_Xte = cell(task_num, 1);
    cv_Yte = cell(task_num, 1);
    
    for t = 1: task_num
        task_sample_size = length(Y{t});
        te_idx = cv_idx : cv_fold : task_sample_size;
        tr_idx = setdiff(1:task_sample_size, te_idx);%返回在A中有，而B中没有的值，结果向量将以升序排序返回。
        cv_Xtr{t} = X{t}(tr_idx, :);
        cv_Ytr{t} = Y{t}(tr_idx, :);
        cv_Xte{t} = X{t}(te_idx, :);
        cv_Yte{t} = Y{t}(te_idx, :);
    end
    %}
    
    for p_idx = 1: length(param_range2)
        %调第一个参数
        %[W,~] = obj_func(cv_Xtr, cv_Ytr, param_range(p_idx),rho,  obj_func_opts);
        %调第二个参数
        [W,~] = obj_func(cv_Xtr, cv_Ytr, rho,param_range2(p_idx),  obj_func_opts);
        perform_mat(p_idx) = perform_mat(p_idx) + eval_func(cv_Yte,cv_Xte, W);%句柄，调用函数
    end
end
perform_mat = perform_mat./cv_fold
perform_mat1=[perform_mat1 perform_mat];
end
%perform_mat = perform_mat./cv_fold
fprintf(']\n')
% if( higher_better)
%     [~,best_idx] = max(perform_mat);
% else
%     [~,best_idx] = min(perform_mat);%返回的是[值，第几列]
% end
%a=max(max(perform_mat1));
A=perform_mat1;
if( higher_better)
    [best_idx1,best_idx2] = find(perform_mat1== max(max(perform_mat1)));
else
     [best_idx1,best_idx2] = find(perform_mat1== max(max(perform_mat1)));
end
best_param1=param_range1(best_idx2);
best_param2=param_range2(best_idx1);
end