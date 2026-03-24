% =========================================================================
% Title: Stable Recovery under Noisy Observations via Standard BPDN
% Description: 
%   This script solves the exact Basis Pursuit De-Noising (BPDN) problem 
%   with an L2-norm constraint as defined in literature:
%   min ||\alpha||_1  s.t.  ||y - H_M * \alpha||_2 <= \eta
%
% =========================================================================

clear variables; clc; close all;

%% 1. Configuration & Hyperparameters
N = 501;
Nc = 50;                     % Monte Carlo iterations (Reduced for CVX runtime)
m_array = 20:20:200;         % Range of sample sizes (m)
SNR_list = [40, 30, 20, 10]; % Signal-to-Noise Ratios in dB
K = 4;                       % Sparsity level
weights = ones(1, N) / N;    % Uniform sampling weights

%% 2. Load Graph & Diffusion Model
% Ensure the selected graph is available in your MATLAB path
load("swGraph.mat"); 

A = full(A);
A = A + eye(N); % Diffusion matrix formulation H = I + \delta A

avErr = zeros(length(m_array), length(SNR_list)); 

%% 3. Main Simulation Loop
% Note: Standard 'for' loop is used. CVX is generally incompatible with 'parfor'.
for snr_idx = 1:length(SNR_list)
    SNR = SNR_list(snr_idx);
    fprintf('\n--- Running SNR = %d dB ---\n', SNR);

    for m_idx = 1:length(m_array)
        m = m_array(m_idx);
        err = zeros(Nc, 1);
        
        fprintf('Computing for m = %d ... ', m);

        for t = 1:Nc 
            %% a. Generate Sparse Seed Signal (\alpha)
            support_set = datasample((1:N), K, 'Replace', false, 'Weights', weights);
            signal_vals = 5 * rand(K, 1);
            alpha_true = zeros(N, 1);    
            alpha_true(support_set) = signal_vals;

            %% b. Uniform Sampling to form H_M
            sampled_indices = datasample((1:N), m, 'Replace', true, 'Weights', weights); 
            sampled_indices = sort(sampled_indices);
            H_M = A(sampled_indices, :); % Measurement Matrix

            %% c. Generate Noisy Observations (y)
            y_clean = H_M * alpha_true; 
            sigPower = norm(y_clean)^2 / m;
            
            if sigPower == 0
                noisePower = 0;
            else
                noisePower = sigPower / (10^(SNR/10));
            end
            
            sigma = sqrt(noisePower);
            e = sigma * randn(m, 1); 
            y = y_clean + e; 

            %% d. Standard BPDN via CVX
            % Tolerance eta = tau * sqrt(m) * sigma, with tau in [1.05, 1.2]
            tau = 1.1; 
            eta = tau * sqrt(m) * sigma;

            cvx_begin quiet
                variable alpha_est(N)
                minimize( norm(alpha_est, 1) )
                subject to
                    % The exact L2 constraint from the paper
                    norm(y - H_M * alpha_est, 2) <= eta
            cvx_end

            %% e. Failsafe and Solution Reconstruction
            if isempty(alpha_est) || any(isnan(alpha_est))
                err(t) = 1; % Penalize severely if solver fails
            else
                % Calculate Relative Error (RE)
                err(t) = norm(alpha_est - alpha_true) / norm(alpha_true);
            end
        end
        
        % Record average error
        avErr(m_idx, snr_idx) = sum(err) / Nc;
        fprintf('Avg Err = %.4f\n', avErr(m_idx, snr_idx));
    end
end

%% 4. Visualization
figure('Position', [100, 100, 600, 500]);
semilogy(m_array, avErr(:, 1), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6); hold on;
semilogy(m_array, avErr(:, 2), 'r-s', 'LineWidth', 1.5, 'MarkerSize', 6);
semilogy(m_array, avErr(:, 3), 'g-^', 'LineWidth', 1.5, 'MarkerSize', 6);
semilogy(m_array, avErr(:, 4), 'k-d', 'LineWidth', 1.5, 'MarkerSize', 6);

grid on; box on;

xlabel('Number of samples', 'FontName', 'Arial', 'FontSize', 12);
ylabel('Relative error', 'FontName', 'Arial', 'FontSize', 12);
set(gca, 'FontName', 'Arial', 'FontSize', 12, 'LineWidth', 1);

title('Stable Recovery via Standard BPDN', 'FontName', 'Arial', 'FontSize', 12);
legend('SNR = 40 dB', 'SNR = 30 dB', 'SNR = 20 dB', 'SNR = 10 dB', ...
       'Location', 'northeast', 'FontName', 'Arial', 'FontSize', 10);