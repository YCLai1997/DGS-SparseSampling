% =========================================================================
% Title: Phase Transition Diagram for Sparse Signal Recovery
% Description: 
%   This script generates a 2D phase transition map by evaluating the 
%   probability of exact recovery across different sparsity levels (k) 
%   and sample sizes (m). It uses a flattened parallel loop (parfor) 
%   to maximize multi-core CPU utilization.
%
% Dependencies:
%   - MATLAB Optimization Toolbox (linprog)
%

% =========================================================================

clear variables; clc; close all;

%% 1. Configuration & Hyperparameters
% System parameters
N = 501;                    % Signal dimension (number of nodes)
Nc = 500;                   % Monte Carlo iterations per grid point
tol_success = 1e-3;         % Relative error threshold for "successful" recovery

% 2D Scan Grid Setup
k_array = 2:2:30;           % Sparsity level range
m_array = 10:10:300;        % Number of samples range
[K_grid, M_grid] = meshgrid(k_array, m_array);
num_configs = numel(K_grid);

weights = ones(1, N) / N;   % Uniform sampling weights

% Optimization parameters for speed
optim_opts = optimoptions('linprog', 'Algorithm', 'dual-simplex', 'Display', 'none');

%% 2. Load Graph Data
% Note: Load your specific graph. Ensure the graph size matches N.
load("ERGraph.mat"); % Make sure this file exists in your path
A = eye(N) + A;      % Add self-loops to form the diffusion matrix H

%% 3. Parallel Grid Search (Phase Transition Computation)
success_rate_flat = zeros(num_configs, 1);

fprintf('Starting parallel computation. Total grid points: %d ...\n', num_configs);

% Flattened parallel loop to maximize CPU efficiency
parfor idx = 1:num_configs
    k_current = K_grid(idx);
    m_current = M_grid(idx);
    
    success_count = 0;
    
    for t = 1:Nc
        % --- 3.1 Signal Generation ---
        support_set = datasample(1:N, k_current, 'Replace', false, 'Weights', weights);
        signal_vals = 5 * rand(k_current, 1);
        
        x_true = zeros(N, 1);    
        x_true(support_set) = signal_vals;
        
        % --- 3.2 Uniform Sampling ---
        sampled_indices = datasample(1:N, m_current, 'Replace', false, 'Weights', weights);
        sampled_indices = sort(sampled_indices);
        
        Phi = A(sampled_indices, :);
        y_meas = Phi * x_true;
        
        % --- 3.3 Fast L1 Recovery ---
        f = ones(2 * N, 1);
        Aeq = [Phi, -Phi];
        beq = y_meas;
        lb = zeros(2 * N, 1);
        
        z_opt = linprog(f, [], [], Aeq, beq, lb, [], optim_opts);
        
        % --- 3.4 Success Evaluation ---
        if ~isempty(z_opt)
            x_recovered = z_opt(1:N) - z_opt(N+1:end);
            rel_err = norm(x_recovered - x_true) / norm(x_true);
            
            if rel_err < tol_success
                success_count = success_count + 1;
            end
        end
    end
    
    % Record success probability for the current (k, m) pair
    success_rate_flat(idx) = success_count / Nc;
    
    % Print progress (order may vary due to parfor, but gives a rough idea)
    if mod(idx, 20) == 0
        fprintf('Completed %d / %d grid points\n', idx, num_configs);
    end
end

fprintf('Computation finished! Generating phase transition diagram...\n');

%% 4. Data Reshaping & Visualization
P_success = reshape(success_rate_flat, size(K_grid));

% Normalize axes (Sparsity ratio and Sampling ratio)
k_ratio = k_array / N;
m_ratio = m_array / N;

figure('Position', [100, 100, 600, 500]);

% Draw heatmap
imagesc(k_ratio, m_ratio, P_success);
set(gca, 'YDir', 'normal'); 
colormap('jet');       
hold on;

% Axis limits (Strictly locked as requested)
xlim([min(k_ratio), max(k_ratio)]);
ylim([0.05, 0.4]); 

% Labels and formatting (Arial font, standard publications style)
xlabel('Sparsity level', 'Interpreter', 'tex', 'FontName', 'Arial', 'FontSize', 14);
ylabel('Sampling ratio', 'Interpreter', 'tex', 'FontName', 'Arial', 'FontSize', 14);

% Colorbar configuration
c = colorbar;
c.Label.String = 'Probability of exact recovery';
c.Label.Interpreter = 'tex';
c.Label.FontName = 'Arial';
c.Label.FontSize = 12;

% General axis aesthetics
set(gca, 'FontName', 'Arial', 'FontSize', 12, 'LineWidth', 1);

