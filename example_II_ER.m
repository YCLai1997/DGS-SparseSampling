% =========================================================================
% Title: Recovery Performance across Different Rewiring Probabilities (b)
% Description: 
%   This script evaluates the exact recovery rate of sparse signals on 
%   Watts-Strogatz small-world graphs (N=10000) under varying rewiring 
%   probabilities (b). It exactly replicates the specified visual legend.
%
% Dependencies:
%   - MATLAB Optimization Toolbox (linprog)
%   - Built-in Graph Theory Toolbox (for wattsStrogatz)
%
% Author: [你的名字/GitHub ID]
% Date: [当前日期]
% =========================================================================

clear variables; clc; close all;

%% 1. Configuration & Hyperparameters
N = 10000;                  % Network size (Large scale!)
K_degree = 40;              % Average degree for the Watts-Strogatz graph
K_sparsity = 4;             % Signal sparsity level
Nc = 20;                    % Monte Carlo iterations (Keep small due to N=10000)
m_array = 50:50:500;        % Range of sample sizes (m)
weights = ones(1, N) / N;   % Uniform sampling weights
tol_success = 1e-3;         % Tolerance for exact recovery

% Rewiring probabilities based on the user's uploaded image
b_array = [0.03, 0.3, 0.7, 0.95, 0.991]; 
num_b = length(b_array);

% Optimization parameters for linprog
optim_opts = optimoptions('linprog', 'Algorithm', 'dual-simplex', 'Display', 'none');

% Initialize result matrix (Rows: m_array, Cols: b_array)
sucRate = zeros(length(m_array), num_b);

%% 2. Visual Style Mapping (Matching the uploaded image exactly)
% Colors approximating standard MATLAB default colormap but matching your image
colors = {
    [0.8500 0.3250 0.0980], % Orange-Red for b=0.03
    [0.4000 0.5000 0.9000], % Soft Blue for b=0.3
    [0.9290 0.6940 0.1250], % Yellow/Gold for b=0.7
    [0.4660 0.6740 0.1880], % Green for b=0.95
    [0.4940 0.1840 0.5560]  % Purple for b=0.991
};
markers = {'^', 'o', 's', 'd', '*'}; % Triangle-up, Circle, Square, Diamond, Asterisk
line_width = 2;
marker_size = 10;

%% 3. Main Simulation Loop
for b_idx = 1:num_b
    b = b_array(b_idx);
    fprintf('\n==================================================\n');
    fprintf('Generating WS Graph (N=%d) with rewiring b = %g ...\n', N, b);
    
    % Generate Watts-Strogatz Graph
    % Note: Use lower case 'w' for newer MATLAB versions: wattsStrogatz
    ws_graph = wattsStrogatz(N, K_degree, b); 
    A = full(adjacency(ws_graph));
    A = A + eye(N); % Diffusion matrix H = I + \delta A
    
    for m_idx = 1:length(m_array)
        m = m_array(m_idx);
        err = zeros(Nc, 1);
        
        fprintf('  Computing for sample size m = %d ... ', m);
        
        % Parallel computation for Monte Carlo runs
        parfor t = 1:Nc
            % a. Generate Sparse Signal
            support_set = datasample((1:N), K_sparsity, 'Replace', false, 'Weights', weights);
            signal_vals = 5 * rand(K_sparsity, 1);
            x_true = zeros(N, 1);
            x_true(support_set) = signal_vals;
            
            % b. Uniform Sampling
            sampled_indices = datasample((1:N), m, 'Replace', false, 'Weights', weights);
            sampled_indices = sort(sampled_indices);
            Phi = A(sampled_indices, :);
            y_meas = Phi * x_true;
            
            % c. Fast Signal Recovery via linprog
            f_obj = ones(2 * N, 1);
            Aeq = [Phi, -Phi];
            beq = y_meas;
            lb = zeros(2 * N, 1);
            
            z_opt = linprog(f_obj, [], [], Aeq, beq, lb, [], optim_opts);
            
            % d. Error Check
            if isempty(z_opt)
                err(t) = 1; % Failsafe
            else
                x_recovered = z_opt(1:N) - z_opt(N+1:end);
                err(t) = norm(x_recovered - x_true) / norm(x_true);
            end
        end
        
        % Calculate Success Rate
        sucRate(m_idx, b_idx) = sum(err < tol_success) / Nc;
        fprintf('Success Rate = %.2f\n', sucRate(m_idx, b_idx));
    end
end

%% 4. Visualization (Replicating the Legend)
figure('Position', [100, 100, 500, 600]);
hold on;

% Plot each line iteratively to apply specific styles
h_plots = zeros(num_b, 1);
legend_strs = cell(num_b, 1);

for b_idx = 1:num_b
    c = colors{b_idx};
    mkr = markers{b_idx};
    
    % Plot lines with thick lines, unfilled markers (except asterisk which has no face)
    if strcmp(mkr, '*')
        h_plots(b_idx) = plot(m_array, sucRate(:, b_idx), ['-', mkr], 'Color', c, ...
            'LineWidth', line_width, 'MarkerSize', marker_size + 2); % Make asterisk slightly larger
    else
        h_plots(b_idx) = plot(m_array, sucRate(:, b_idx), ['-', mkr], 'Color', c, ...
            'LineWidth', line_width, 'MarkerSize', marker_size, 'MarkerFaceColor', 'none', 'MarkerEdgeColor', c);
    end
    
    legend_strs{b_idx} = sprintf('b=%g', b_array(b_idx));
end

grid on; box on;

% Formatting axes
xlabel('Number of samples', 'FontName', 'Arial', 'FontSize', 14);
ylabel('Probability of exact recovery', 'FontName', 'Arial', 'FontSize', 14);
set(gca, 'FontName', 'Arial', 'FontSize', 12, 'LineWidth', 1);
ylim([-0.05, 1.05]);

% Replicate the exact legend
lgd = legend(h_plots, legend_strs, 'Location', 'southeast', 'FontName', 'Arial', 'FontSize', 14);
% lgd.ItemTokenSize = [30, 18]; % Optional: make legend lines longer if needed