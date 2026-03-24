function W = normalize_rows(A, norm_type)
% =========================================================================
% Normalize the rows of a matrix.
% 
% Inputs:
%   A         - Input matrix (N x M)
%   norm_type - 'L1' (default): Elements in each row sum to 1. 
%               'L2'          : The L2-norm of each row equals 1.
%
% Outputs:
%   W         - Row-normalized matrix
%
% Author: [你的名字/GitHub ID]
% =========================================================================

    % 如果没有指定归一化类型，默认使用 L1 (概率归一化)
    if nargin < 2
        norm_type = 'L1';
    end

    % 获取矩阵的尺寸
    [num_rows, ~] = size(A);
    W = zeros(size(A));

    if strcmpi(norm_type, 'L1')
        % --- L1 归一化：使每行元素之和等于 1 ---
        % 常用于生成随机漫步(Random Walk)的转移概率矩阵
        
        row_sums = sum(A, 2);
        
        % 防御性编程：防止出现全0行导致除以0出现 NaN
        % 将和为 0 的行的分母强制设为 1（这样 0/1 依然是 0）
        row_sums(row_sums == 0) = 1; 
        
        % 利用 MATLAB 的隐式扩展(Broadcasting)进行极速矩阵除法
        W = A ./ row_sums;

    elseif strcmpi(norm_type, 'L2')
        % --- L2 归一化：使每行向量的 L2 范数（长度）等于 1 ---
        % 常用于计算余弦相似度 (Cosine Similarity)
        
        row_norms = sqrt(sum(A.^2, 2)); % 或者直接用 vecnorm(A, 2, 2)
        
        % 防御性编程：防止除以零
        row_norms(row_norms == 0) = 1;
        
        W = A ./ row_norms;

    else
        error('Unsupported normalization type. Please use ''L1'' or ''L2''.');
    end
end