% =========================================================================
% Title: Noiseless Stable Recovery on Watts-Strogatz Small-World Graphs
% Description: 
%   This script evaluates the performance of L1-minimization for signal 
%   recovery on a Watts-Strogatz small-world graph. It calculates the 
%   relative recovery error, observation error, and exact recovery success 
%   rate across varying sample sizes (m).
%
% Dependencies:
%   - MATLAB Optimization Toolbox (linprog)
%   - Built-in Graph Theory Toolbox (for WattsStrogatz)
%
% Author: [你的名字/GitHub ID]
% Date: [当前日期]
% =========================================================================

clear variables; clc; close all;

%% 1. Configuration & Hyperparameters
N = 2001;                   % Signal dimension (number of nodes)
K = 4;                      % Sparsity level
Nc = 500;                    % Monte Carlo iterations (Increased to 50 for smoother curves)
m_array = 10:10:300;        % Sample size array (Step by 10 for efficiency)
weights = ones(1, N) / N;   % Uniform sampling weights
tol_success = 0.01;         % Threshold for determining "successful" recovery

% Fast Linear Programming Options
optim_opts = optimoptions('linprog', 'Algorithm', 'dual-simplex', 'Display', 'none');

%% 2. Graph Generation (Watts-Strogatz)
fprintf('Generating Watts-Strogatz graph (N=%d) ...\n', N);
ws_graph = WattsStrogatz(N, 41, 0.1); 
A = full(adjacency(ws_graph));
A = A + eye(N);             % Form diffusion matrix: H = I + \delta A

%% 3. Main Parallel Simulation Loop
avErr   = zeros(length(m_array), 1);
avErr1  = zeros(length(m_array), 1);
sucRate = zeros(length(m_array), 1);

for m_idx = 1:length(m_array)
    m = m_array(m_idx);
    
    % Pre-allocate arrays for parfor loop
    err  = zeros(Nc, 1);
    err1 = zeros(Nc, 1);
    
    fprintf('Running for sample size m = %d ...\n', m);
    
    parfor t = 1:Nc
        %% a. Generate Sparse Seed Signal
        support_set = datasample((1:N), K, 'Replace', false, 'Weights', weights);
        signal_vals = 5 * rand(K, 1);
        
        x_true = zeros(N, 1);
        x_true(support_set) = signal_vals;
        y_global = A * x_true; % Global state vector
        
        %% b. Uniform Sampling
        sampled_indices = datasample((1:N), m, 'Replace', false, 'Weights', weights);
        sampled_indices = sort(sampled_indices);
        
        Phi = A(sampled_indices, :);
        y_meas = Phi * x_true;
        
        %% c. Fast Signal Recovery via L1 Minimization
        % Standard LP Formulation: min sum(u) + sum(v) s.t. [Phi, -Phi]*[u;v] = y_meas
        f_obj = ones(2 * N, 1);
        Aeq = [Phi, -Phi];
        beq = y_meas;
        lb = zeros(2 * N, 1);
        
        z_opt = linprog(f_obj, [], [], Aeq, beq, lb, [], optim_opts);
        
        if isempty(z_opt)
            x_recovered = zeros(N, 1); % Failsafe
        else
            x_recovered = z_opt(1:N) - z_opt(N+1:end);
        end
        
        %% d. Error Calculation
        % Relative recovery error
        err(t) = norm(x_recovered - x_true) / norm(x_true);
        % Observation error based on global state
        err1(t) = norm(A * x_recovered - y_global) / norm(y_global);
    end
    
    % Aggregate results safely outside the parfor loop
    avErr(m_idx)  = sum(err) / Nc;
    avErr1(m_idx) = sum(err1) / Nc;
    sucRate(m_idx) = sum(err < tol_success) / Nc; % Safe calculation of success rate
end

fprintf('Simulation complete! Plotting results...\n');

%% 4. Visualization (Dual Y-Axis for Error and Success Rate)
figure('Position', [100, 100, 600, 450]);

% Left Y-Axis: Recovery Error
yyaxis left;
plot(m_array, avErr, 'b.-', 'LineWidth', 1.5, 'MarkerSize', 15);
ylabel('Relative Recovery Error', 'FontName', 'Arial', 'FontSize', 12, 'Color', 'b');
set(gca, 'YColor', 'b');
ylim([0, 1]); % Force error to scale elegantly between 0 and 1 (if appropriate)

% Right Y-Axis: Success Rate
yyaxis right;
plot(m_array, sucRate, 'r.-', 'LineWidth', 1.5, 'MarkerSize', 15);
ylabel('Success Rate', 'FontName', 'Arial', 'FontSize', 12, 'Color', 'r');
set(gca, 'YColor', 'r');
ylim([0, 1.05]); % Success rate is strictly 0 to 1

% Common Axis Formatting
grid on; box on;
xlabel('Number of samples', 'FontName', 'Arial', 'FontSize', 12);
title('Recovery Performance on Watts-Strogatz Graph', 'FontName', 'Arial', 'FontSize', 12);
set(gca, 'FontName', 'Arial', 'FontSize', 12, 'LineWidth', 1);

legend({'Recovery Error', 'Success Rate'}, 'Location', 'east', 'FontName', 'Arial', 'FontSize', 10);