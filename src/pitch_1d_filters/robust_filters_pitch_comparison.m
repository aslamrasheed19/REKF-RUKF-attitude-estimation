% Clear workspace
clear; clc; close all;

%% 1. Load Data
filename = 'pitch_data.csv';
if ~isfile(filename)
    error('File %s not found. Please ensure it is in the current directory.', filename);
end

data = readtable(filename);
time = data.Time;
z_meas = data.Sensor_Pitch;
true_pitch = data.True_Pitch;

N = length(time);
dt = 0.1; % Sampling time

% Optional: Add artificial outliers to clearly see the Robust difference
% z_meas(150:155) = z_meas(150:155) + 5.0; 
% z_meas(350:355) = z_meas(350:355) - 4.0;

%% 2. Filter Configuration
% State: [pitch; velocity]
n_states = 2;
n_meas = 1;

% Initial Conditions
x0 = [0; 0];
P0 = eye(n_states) * 1;

% Process Noise (Q)
sigma_acc = 0.5;
G = [0.5*dt^2; dt];
Q = G * G' * sigma_acc^2;

% Measurement Noise (R)
R_std = 0.5; 

% Robustness Parameters
k_huber = 2.0; % Threshold for outlier detection (sigmas)

% UKF Parameters
alpha = 1e-3;
kappa = 0;
beta = 2;
lambda = alpha^2 * (n_states + kappa) - n_states;
w_m0 = lambda / (n_states + lambda);
w_c0 = w_m0 + (1 - alpha^2 + beta);
W_m = [w_m0, repmat(1/(2*(n_states + lambda)), 1, 2*n_states)];
W_c = [w_c0, repmat(1/(2*(n_states + lambda)), 1, 2*n_states)];

%% 3. Initialization of All Filters
% EKF / REKF
x_ekf = x0; P_ekf = P0;
x_rekf = x0; P_rekf = P0;

% UKF / RUKF
x_ukf = x0; P_ukf = P0;
x_rukf = x0; P_rukf = P0;

% Storage
hist_ekf = zeros(N, n_states);
hist_rekf = zeros(N, n_states);
hist_ukf = zeros(N, n_states);
hist_rukf = zeros(N, n_states);
mnsf_rekf = zeros(N, 1); % Measurement Noise Scaling Factor for REKF
mnsf_rukf = zeros(N, 1); % Measurement Noise Scaling Factor for RUKF

%% 4. Main Loop
% System Matrices for (Linear) Kalman Filter
F_sys = [1, dt; 0, 1]; 
H_sys = [1, 0];

for k = 1:N
    z = z_meas(k);
    
    % ==========================================
    % A. Extended Kalman Filter (EKF) - Linear CV
    % ==========================================
    % Predict
    x_pred = F_sys * x_ekf;
    P_pred = F_sys * P_ekf * F_sys' + Q;
    
    % Update
    y = z - H_sys * x_pred;
    S = H_sys * P_pred * H_sys' + R_std;
    K = P_pred * H_sys' / S;
    
    x_ekf = x_pred + K * y;
    P_ekf = (eye(n_states) - K * H_sys) * P_pred;
    hist_ekf(k,:) = x_ekf';
    
    % ==========================================
    % B. Robust EKF (REKF)
    % ==========================================
    % Predict
    x_pred = F_sys * x_rekf;
    P_pred = F_sys * P_rekf * F_sys' + Q;
    
    % Robust Update
    y = z - H_sys * x_pred;
    S_pre = H_sys * P_pred * H_sys' + R_std;
    
    % Check Innovation (Mahalanobis Distance)
    mah_dist = abs(y) / sqrt(S_pre);
    
    % Calculate MNSF (Measurement Noise Scaling Factor)
    if mah_dist > k_huber
        mnsf = (mah_dist / k_huber)^2; % Inflate R
    else
        mnsf = 1.0;
    end
    mnsf_rekf(k) = mnsf;
    
    R_eff = R_std * mnsf;
    S = H_sys * P_pred * H_sys' + R_eff;
    K = P_pred * H_sys' / S;
    
    x_rekf = x_pred + K * y;
    P_rekf = (eye(n_states) - K * H_sys) * P_pred;
    hist_rekf(k,:) = x_rekf';
    
    % ==========================================
    % C. Unscented Kalman Filter (UKF)
    % ==========================================
    [x_ukf, P_ukf] = ukf_step(x_ukf, P_ukf, z, Q, R_std, dt, W_m, W_c, n_states, lambda, 1.0);
    hist_ukf(k,:) = x_ukf';
    
    % ==========================================
    % D. Robust UKF (RUKF)
    % ==========================================
    % We modify the standard UKF function to return innovation stats and accept scaling
    % To keep code clean, we implement a specific RUKF step here or reuse the function carefully.
    % Let's do the prediction first to check innovation.
    
    % 1. Generate Sigma Points
    S_root = chol( (n_states + lambda) * P_rukf )';
    X_sig = [x_rukf, x_rukf + S_root, x_rukf - S_root];
    
    % 2. Predict State
    X_sig_pred = F_sys * X_sig; % Linear propagation
    x_pred = sum(W_m .* X_sig_pred, 2);
    P_pred = Q;
    for i = 1:(2*n_states+1)
        diff = X_sig_pred(:,i) - x_pred;
        P_pred = P_pred + W_c(i) * (diff * diff');
    end
    
    % 3. Predict Measurement
    % Redraw sigma points from P_pred (Standard UKF usually does this)
    S_root_pred = chol( (n_states + lambda) * P_pred )';
    X_sig_pred_new = [x_pred, x_pred + S_root_pred, x_pred - S_root_pred];
    Z_sig_pred = H_sys * X_sig_pred_new;
    
    z_pred = sum(W_m .* Z_sig_pred, 2);
    S_pred = 0; % Will add R later
    for i = 1:(2*n_states+1)
        diff = Z_sig_pred(:,i) - z_pred;
        S_pred = S_pred + W_c(i) * (diff * diff');
    end
    
    % 4. Check Innovation for Robustness
    y = z - z_pred;
    S_total_pre = S_pred + R_std;
    mah_dist = abs(y) / sqrt(S_total_pre);
    
    if mah_dist > k_huber
        mnsf = (mah_dist / k_huber)^2;
    else
        mnsf = 1.0;
    end
    mnsf_rukf(k) = mnsf;
    
    % 5. Update
    S_total = S_pred + (R_std * mnsf); % Inflate R
    
    % Cross Covariance
    Pxz = zeros(n_states, n_meas);
    for i = 1:(2*n_states+1)
        diff_x = X_sig_pred_new(:,i) - x_pred;
        diff_z = Z_sig_pred(:,i) - z_pred;
        Pxz = Pxz + W_c(i) * (diff_x * diff_z');
    end
    
    K = Pxz / S_total;
    x_rukf = x_pred + K * y;
    P_rukf = P_pred - K * S_total * K';
    
    hist_rukf(k,:) = x_rukf';
end

%% 5. Plotting

% --- Figure 1: EKF vs REKF ---
figure('Color', 'w', 'Position', [100, 100, 900, 700]);

subplot(3,1, [1 2]); % Main plot takes top 2/3
plot(time, z_meas, 'r.', 'MarkerSize', 6, 'DisplayName', 'Measurements'); hold on;
plot(time, true_pitch, 'k', 'LineWidth', 2, 'DisplayName', 'True Pitch');
plot(time, hist_ekf(:,1), 'g--', 'LineWidth', 1.5, 'DisplayName', 'EKF');
plot(time, hist_rekf(:,1), 'b-', 'LineWidth', 1.5, 'DisplayName', 'REKF');
title('Figure 1: EKF vs. Robust EKF');
ylabel('Pitch (rad)');
legend('Location', 'best');
grid on;
ylim([min(true_pitch)-1, max(true_pitch)+1]);

subplot(3,1,3); % MNSF plot
plot(time, mnsf_rekf, 'r', 'LineWidth', 1.5);
title('MNSF of REKF (Measurement Noise Scaling Factor)');
xlabel('Time (s)');
ylabel('Scaling Factor');
grid on;


% --- Figure 2: UKF vs RUKF ---
figure('Color', 'w', 'Position', [150, 150, 900, 700]);

subplot(3,1, [1 2]); 
plot(time, hist_ukf(:,1), 'g--', 'LineWidth', 1.5, 'DisplayName', 'UKF'); hold on;
plot(time, true_pitch, 'k', 'LineWidth', 2, 'DisplayName', 'True Pitch');
plot(time, z_meas, 'r.', 'MarkerSize', 6, 'DisplayName', 'Measurements');
plot(time, hist_rukf(:,1), 'm-', 'LineWidth', 1.5, 'DisplayName', 'RUKF');
title('Figure 2: UKF vs. Robust UKF');
ylabel('Pitch (rad)');
legend('Location', 'best');
grid on;
ylim([min(true_pitch)-1, max(true_pitch)+1]);

subplot(3,1,3); % MNSF plot
plot(time, mnsf_rukf, 'm', 'LineWidth', 1.5);
title('MNSF of RUKF (Measurement Noise Scaling Factor)');
xlabel('Time (s)');
ylabel('Scaling Factor');
grid on;

% Figure: REKF vs RUKF
figure('Color', 'w', 'Position', [150, 150, 900, 700]); hold on;
plot(time, hist_rekf(:,1), 'm-', 'LineWidth', 2.5, 'DisplayName', 'REKF'); 
plot(time, hist_rukf(:,1), 'g-', 'LineWidth', 1.5, 'DisplayName', 'RUKF');
plot(time, z_meas, 'r--', 'LineWidth', 1, 'DisplayName', 'Measurements');
title('Figure 2: Robust EKF vs. Robust UKF');
ylabel('Pitch (rad)');
legend('Location', 'best');
grid on;

%% Helper Function: Standard UKF Step
function [x_new, P_new] = ukf_step(x, P, z, Q, R, dt, Wm, Wc, n, lambda, mnsf)
    % Linear System Matrices (Hardcoded for this problem)
    F = [1, dt; 0, 1];
    H = [1, 0];

    % 1. Sigma Points
    S_root = chol( (n + lambda) * P )';
    X_sig = [x, x + S_root, x - S_root];
    
    % 2. Prediction
    X_sig_pred = F * X_sig;
    x_pred = sum(Wm .* X_sig_pred, 2);
    P_pred = Q;
    for i = 1:length(Wm)
        d = X_sig_pred(:,i) - x_pred;
        P_pred = P_pred + Wc(i) * (d * d');
    end
    
    % 3. Update
    % Resample sigmas
    S_root_p = chol( (n + lambda) * P_pred )';
    X_sig_p = [x_pred, x_pred + S_root_p, x_pred - S_root_p];
    Z_sig = H * X_sig_p;
    
    z_pred = sum(Wm .* Z_sig, 2);
    S = R * mnsf; % Apply scaling if any (standard UKF mnsf=1)
    for i = 1:length(Wm)
        d = Z_sig(:,i) - z_pred;
        S = S + Wc(i) * (d * d');
    end
    
    Pxz = zeros(n, 1);
    for i = 1:length(Wm)
        dx = X_sig_p(:,i) - x_pred;
        dz = Z_sig(:,i) - z_pred;
        Pxz = Pxz + Wc(i) * (dx * dz');
    end
    
    K = Pxz / S;
    x_new = x_pred + K * (z - z_pred);
    P_new = P_pred - K * S * K';
end