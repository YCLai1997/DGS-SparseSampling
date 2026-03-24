% =========================================================================
% Title: Signal Recovery on doubly stochastic  matrix (Uniform vs. Variable Density)
% Description: 
%   This script compares the recovery error of L1-minimization using 
%   standard Uniform sampling versus a novel Variable Density sampling 
%   strategy on doubly stochastic  matrix.
%
% =========================================================================

clear variables; clc; close all;
N=1000;
load("doublyStochastic.mat")
optim_opts = optimoptions('linprog', 'Algorithm', 'dual-simplex', 'Display', 'none');
K_sparsity = 20;     % Signal sparsity level
Nc = 300;           % Monte Carlo iterations
m_list = 200:20:500; % Range of sample sizes (m)

%% 2. Sampling Weights Calculation
fprintf('Calculating Variable Density Probabilities...\n');

% --- Uniform Sampling Weights ---
prob_uni = ones(N, 1) / N;

% --- Variable Density Sampling Weights (Theorem 4) ---
% Calculate Gamma: Gamma = N * (H'H)^-1
HHt = H' * H;
Gamma = N * (HHt \eye(N) ); 
M_mat = H * Gamma;
% Calculate local coherence parameters
phi_max = max(abs(H), [], 2);        % Max absolute value per row of H
t_phi_max = max(abs(M_mat), [], 2);  % Max absolute value per row of Gamma

% Calculate final weights and normalize to probabilities
weights_var = sqrt(phi_max .* phi_max);
prob_var = weights_var / sum(weights_var); 


%% 3. Main Simulation Loop (Parallelized)
avErr_uni = zeros(length(m_list), 1);
avErr_var = zeros(length(m_list), 1);

 
for m_idx = 1:length(m_list)
    m = m_list(m_idx);
    
    % Arrays to hold errors for parfor loop
    err_u_arr = zeros(Nc, 1);
    err_v_arr = zeros(Nc, 1);
    
    fprintf('Sampling m=%d ... ', m);
    
    
     for t = 1:Nc
        % --- Generate Sparse Signal ---
        support_set = randperm(N, K_sparsity);
        alpha_true = zeros(N, 1);
        alpha_true(support_set) = randn(K_sparsity, 1); % Gaussian distribution
        
        % ===========================
        % Strategy 1: Uniform Sampling
        % ===========================
        % 'true' means with replacement, matching your original logic
        idx_uni = randsample(N, m, true, prob_uni); 
        idx_uni = sort(unique(idx_uni)); 
        Phi_u = H(idx_uni, :); 
        y_u = Phi_u * alpha_true;   
        
        % Fast BP Recovery via linprog
        f_obj = ones(2 * N, 1);
        lb = zeros(2 * N, 1);
        z_u = linprog(f_obj, [], [], [Phi_u, -Phi_u], y_u, lb, [], optim_opts);
        if ~isempty(z_u)
            alpha_rec_u = z_u(1:N) - z_u(N+1:end);
            err_u_arr(t) = norm(alpha_rec_u - alpha_true) / norm(alpha_true);
        else
            err_u_arr(t) = 1; % Failsafe
        end
        
        % ===========================
        % Strategy 2: Variable Density
        % ===========================
        % 使用 datasample 代替 randsample
        idx_v = datasample(1:N, m, 'Replace', false, 'Weights', prob_var);
        idx_v = sort(idx_v);
        Phi_v = H(idx_v, :); 
        y_v = Phi_v * alpha_true;   

        % y_v = Phi_v * alpha_true;
        
        % Fast BP Recovery via linprog
        z_v = linprog(f_obj, [], [], [Phi_v, -Phi_v], y_v, lb, [], optim_opts);
        if ~isempty(z_v)
            alpha_rec_v = z_v(1:N) - z_v(N+1:end);
            err_v_arr(t) = norm(alpha_rec_v - alpha_true) / norm(alpha_true);
        else
            err_v_arr(t) = 1; % Failsafe
        end
    end
    
    % Calculate average errors
    avErr_uni(m_idx) = sum(err_u_arr) / Nc;
    avErr_var(m_idx) = sum(err_v_arr) / Nc;
    
    fprintf('Uni Err: %.4f | Var Err: %.4f\n', avErr_uni(m_idx), avErr_var(m_idx));
end

%% 4. Visualization
% Fallback to standard MATLAB high-contrast colors if GetColors() is missing
color_uni = [0.9290, 0.6940, 0.1250]; % Yellowish
color_var = [0.0000, 0.4470, 0.7410]; % Blueish

% If you have your custom function, uncomment the next lines:
% [all_themes, all_colors] = GetColors(); 
% color_uni = all_colors(3, :);
% color_var = all_colors(1, :);

figure('Position', [100, 100, 600, 500]);
plot(m_list, avErr_uni, '-o', 'Color', color_uni, 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'w'); hold on;
plot(m_list, avErr_var, '-^', 'Color', color_var, 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'w'); 

grid on; box on;
legend('Random sampling', 'Variable density sampling', 'FontSize', 12, 'Location', 'northeast');

xlabel('Number of samples', 'FontName', 'Arial', 'FontSize', 12);
ylabel('Recovery error', 'FontName', 'Arial', 'FontSize', 12);
title('Recovery Performance on SBM Graph', 'FontName', 'Arial', 'FontSize', 12);
set(gca, 'FontName', 'Arial', 'FontSize', 12, 'LineWidth', 1);