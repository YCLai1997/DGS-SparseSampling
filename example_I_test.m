% =========================================================================
% Title: Noiseless Stable Recovery via L1 Minimization (Theorem 1)
% Description: 
%   This script simulates the noiseless stable recovery of sparse signals 
%   on various graph structures. It utilizes an optimized linear programming 
%   (linprog) approach for fast L1-norm minimization.
% 
% Dependencies:
%   - MATLAB Optimization Toolbox (for linprog)
%

% =========================================================================

clear variables; clc; close all;

%% 1. Configuration & Hyperparameters
% System parameters
N = 501;                % Signal dimension (number of nodes)
K = 4;                  % Sparsity level (number of non-zero elements)
Nc = 500;               % Number of Monte Carlo iterations
m_array = 10:10:500;    % Array of sample sizes (m)
weights = ones(1, N) / N; % Uniform sampling weights

% Optimization parameters (Pre-configured for speed)
% Using dual-simplex for fast resolution of L1 minimization cast as LP
optim_opts = optimoptions('linprog', 'Algorithm', 'dual-simplex', 'Display', 'none');

%% 2. Load Graph Data (Adjacency Matrix A)
% Uncomment the specific graph model you wish to test. 
% Ensure the corresponding .mat file is in your MATLAB path.

% load("RegGraph.mat"); 
% load("StarGraph.mat"); 
% load("StarGraph_REG.mat"); 
% load("BAGraph.mat"); 
% load("swGraph.mat");
load("sbm_graph.mat");  % Currently active graph


%% 3. Main Simulation Loop
avErr = zeros(length(m_array), 1);

for m_idx = 1:length(m_array)
    m = m_array(m_idx);
    err = zeros(Nc, 1);
    
    fprintf('Computing for sample size m = %d ...\n', m);
    
    parfor t = 1:Nc
        % --- 3.1 Generate Sparse Seed Signal ---
        % Randomly select K support nodes
        support_set = datasample(1:N, K, 'Replace', false, 'Weights', weights);
        signal_vals = 5 * rand(K, 1);
        
        x_true = zeros(N, 1);    
        x_true(support_set) = signal_vals;
        
        % --- 3.2 Uniform Sampling ---
        % Sample m nodes without replacement
        sampled_indices = datasample(1:N, m, 'Replace', false, 'Weights', weights);
        sampled_indices = sort(sampled_indices);
        
        % Construct measurement matrix and observation vector
        Phi = A(sampled_indices, :);
        y_meas = Phi * x_true;
        
        % --- 3.3 Fast Signal Recovery via L1 Minimization ---
        % Cast L1 minimization to standard Linear Programming:
        % Let x = u - v, where u >= 0, v >= 0
        % min sum(u) + sum(v) s.t. [Phi, -Phi]*[u; v] = y_meas
        f = ones(2 * N, 1);
        Aeq = [Phi, -Phi];
        beq = y_meas;
        lb = zeros(2 * N, 1);
        
        % Solve using optimized linprog
        z = linprog(f, [], [], Aeq, beq, lb, [], optim_opts);
        
        if isempty(z)
            x_recovered = zeros(N, 1); % Failsafe for extreme ill-conditioned cases
        else
            x_recovered = z(1:N) - z(N+1:end);
        end
        
        % Calculate relative error
        err(t) = norm(x_recovered - x_true) / norm(x_true);
    end
    
    avErr(m_idx) = sum(err) / Nc;
end

%% 4. Visualization
figure('Position', [100, 100, 500, 400]);
plot(m_array, avErr, 'k.-', 'LineWidth', 1.5, 'MarkerSize', 15);
grid on;
xlabel('Number of samples $m$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Relative Error (RE)', 'Interpreter', 'latex', 'FontSize', 12);
title('Noiseless Recovery Performance', 'FontSize', 12);

% Optional: Save figure to file
% saveas(gcf, 'Recovery_Performance.png');