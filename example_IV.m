% =========================================================================
% Title: Stable Recovery on Minnesota Road Network (Uniform vs. Variable Density)
% Description: 
%   This script evaluates L1-minimization signal recovery on a real-world 
%   graph (Minnesota Road Network). It compares standard Uniform Random 
%   Sampling against a deterministic Variable Density Sampling strategy.
%   The diffusion matrix is constructed strictly as a Doubly Stochastic 
%   Matrix using the Metropolis-Hastings rule.
% =========================================================================

clear variables; clc; close all;

%% 0. Check GSPBox Dependency
try
    G = gsp_minnesota(); 
catch
    error('GSPBox not found! Please download GSPBox and run gsp_start.m first.');
end

%% 1. Parameters Setup & Load Minnesota Network
fprintf('Loading Minnesota Road Network...\n');
A_bin = full(G.W);  % Binary Adjacency Matrix
N = G.N;            % Number of nodes (2642)
K_sparsity = 5;     % Sparsity (e.g., 5 gas leak sources)
Nc = 1000;          % Monte Carlo iterations
m_list = 200:100:1000; % Range of sample sizes

% Optimization Options for fast LP
optim_opts = optimoptions('linprog', 'Algorithm', 'dual-simplex', 'Display', 'none');

%% 2. Construct Doubly Stochastic Matrix H (Metropolis-Hastings)
fprintf('Constructing strictly Doubly Stochastic Matrix...\n');
degs = sum(A_bin, 2);
A = zeros(N, N);

% Assign Metropolis weights strictly to existing edges
[rows, cols] = find(triu(A_bin, 1)); 
for i_edge = 1:length(rows)
    u = rows(i_edge); 
    v = cols(i_edge);
    weight = 1 / max(degs(u), degs(v));
    A(u, v) = weight;
    A(v, u) = weight;
end

% Fill diagonal to ensure row sums (and column sums) strictly equal 1
for i = 1:N
    A(i, i) = 1 - sum(A(i, :));
end

delta = 1; 
H = eye(N) + delta * A; % Diffusion model

%% 3. Pre-compute Sampling Weights (Offline Structural Analysis)
fprintf('Computing Structural Matrix Gamma and Sampling Probabilities...\n');

% --- Variable Density Sampling Weights (Theorem 4) ---
HHt = H' * H;
% Correct Inverse: (HHt \ eye(N)) is mathematically inv(HHt)
Gamma = N * (HHt \ eye(N)); 
M_mat = H * Gamma;

% Local coherence parameters
phi_max = max(abs(H), [], 2);        
t_phi_max = max(abs(M_mat), [], 2); 

% Calculate weights and normalize to probabilities
weights_var = sqrt(phi_max .* t_phi_max);
prob_var = weights_var / sum(weights_var); 

% Deterministic pre-selection: sort nodes by priority
[~, sorted_nodes_by_prob] = sort(prob_var, 'descend');

%% 4. Main Parallel Simulation Loop
avErr_uni = zeros(length(m_list), 1);
avErr_var = zeros(length(m_list), 1);

fprintf('Starting Parallel Noiseless Simulation...\n');

for m_idx = 1:length(m_list)
    m = m_list(m_idx);
    
    % Arrays to hold parallel results
    err_u_arr = zeros(Nc, 1);
    err_v_arr = zeros(Nc, 1);
    
    % Deterministic Strategy: Fix the top 'm' important nodes
    idx_var = sorted_nodes_by_prob(1:m);
    Phi_v = H(idx_var, :);
    
    fprintf('Sampling m=%d (Ratio: %.2f) ...\n', m, m/N);
    
    % Parfor for massive speedup
    parfor t = 1:Nc
        % --- Generate Sparse Source Signal ---
        support_set = randperm(N, K_sparsity);
        alpha_true = zeros(N, 1);
        alpha_true(support_set) = randn(K_sparsity, 1); 
        
        % True physical diffusion state
        y_global = H * alpha_true;
        
        % =======================================================
        % Strategy 1: Uniform Random Sampling (Exact m nodes)
        % =======================================================
        % randperm is the fastest & safest way to get exactly m unique nodes
        idx_uni = randperm(N, m); 
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
            err_u_arr(t) = 1;
        end
        
        % =======================================================
        % Strategy 2: Variable Density Sampling
        % =======================================================
        y_v = Phi_v * alpha_true; 
        
        z_v = linprog(f_obj, [], [], [Phi_v, -Phi_v], y_v, lb, [], optim_opts);
        if ~isempty(z_v)
            alpha_rec_v = z_v(1:N) - z_v(N+1:end);
            err_v_arr(t) = norm(alpha_rec_v - alpha_true) / norm(alpha_true);
        else
            err_v_arr(t) = 1;
        end
    end
    
    % Average the errors
    avErr_uni(m_idx) = sum(err_u_arr) / Nc;
    avErr_var(m_idx) = sum(err_v_arr) / Nc;
    
    fprintf('  -> Uni Err: %.4f | Var Err: %.4f\n', avErr_uni(m_idx), avErr_var(m_idx));
end

%% 5. Visualization (Strictly matched to publication format)
% Fallback colors if GetColors() is missing
color_var = [0.0000, 0.4470, 0.7410]; % Blueish
color_uni = [0.9290, 0.6940, 0.1250]; % Yellowish

% Uncomment these lines if you have your custom GetColors() function
% [~, all_colors] = GetColors();  
% color_var = all_colors(1, :);
% color_uni = all_colors(3, :);

figure('Color', 'w', 'Position', [100, 100, 600, 450]);

plot(m_list, avErr_var, '-^', 'MarkerIndices', 1:1:length(m_list), ...
    'Color', color_var, 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'w');
hold on;
plot(m_list, avErr_uni, '-o', 'MarkerIndices', 1:1:length(m_list), ...
    'Color', color_uni, 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'w');

legend('Variable density sampling', 'Uniform random sampling', ...
       'FontSize', 12, 'Location', 'northeast');

xlabel('Number of samples', 'FontName', 'Arial', 'FontSize', 12);
ylabel('Recovery error', 'FontName', 'Arial', 'FontSize', 12);
grid on;

set(gca, 'FontSize', 12, 'LineWidth', 1, 'FontName', 'Arial');