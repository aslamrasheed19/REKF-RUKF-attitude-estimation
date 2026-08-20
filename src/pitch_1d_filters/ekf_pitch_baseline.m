% Clear workspace and figures
clear; clc; close all;

%% 1. Load and Preprocess Data
% Ensure pitch_data.csv is in the current folder
filename = 'pitch_data.csv';
if ~isfile(filename)
    error('File %s not found in the current directory.', filename);
end

data = readtable(filename);

time = data.Time;
z_meas = data.Sensor_Pitch; % Noisy measurements
true_pitch = data.True_Pitch; % Ground truth for validation

N = length(time);
dt = 0.1; % Sampling time (derived from data)

%% 2. EKF Initialization
% State vector: [pitch_angle; pitch_velocity]
n_states = 2;
x_est = [0; 0]; % Initial state estimate (angle=0, vel=0)

% Initial State Covariance Matrix (P)
P = eye(n_states) * 1; 

% Measurement Noise Covariance (R)
% Calculated variance from data is approx 0.5
R = 0.5; 

% Process Noise Covariance (Q)
% Allows the velocity to change (acceleration noise)
% Using a discrete Wiener process noise model
sigma_a = 0.5; % Acceleration noise standard deviation (tunable)
G = [0.5*dt^2; dt];
Q = G * G' * sigma_a^2;

% Storage for plotting
x_history = zeros(N, n_states);
P_history = zeros(N, n_states);

%% 3. EKF Loop
% Define Jacobian matrices (Linear for CV model, but structure is general for EKF)
% State Transition Jacobian (F)
F = [1, dt; 
     0, 1];

% Measurement Jacobian (H)
H = [1, 0]; 

for k = 1:N
    % --- Prediction Step ---
    % Project the state ahead
    % f(x) = [angle + vel*dt; vel]
    x_pred = F * x_est; 
    
    % Project the error covariance ahead
    P_pred = F * P * F' + Q;
    
    % --- Update Step ---
    % Measurement at time k
    z = z_meas(k);
    
    % Measurement Residual (Innovation)
    % h(x) = angle
    z_pred = H * x_pred;
    y = z - z_pred;
    
    % Innovation Covariance
    S = H * P_pred * H' + R;
    
    % Optimal Kalman Gain
    K = P_pred * H' / S;
    
    % Update State Estimate
    x_est = x_pred + K * y;
    
    % Update Covariance Estimate
    P = (eye(n_states) - K * H) * P_pred;
    
    % --- Store Data ---
    x_history(k, :) = x_est';
end

%% 4. Visualization and Analysis
figure('Color', 'w', 'Position', [100, 100, 800, 600]);

% Plot 1: Pitch Angle Tracking
subplot(2,1,1);
plot(time, z_meas, 'g.', 'DisplayName', 'Sensor Measurement'); hold on;
plot(time, true_pitch, 'k--', 'LineWidth', 1.5, 'DisplayName', 'True Pitch');
plot(time, x_history(:,1), 'b-', 'LineWidth', 2, 'DisplayName', 'EKF Estimate');
title('EKF Pitch Estimation');
xlabel('Time (s)');
ylabel('Pitch (rad)');
legend('Location', 'best');
grid on;

% Plot 2: Error Analysis
subplot(2,1,2);
err_meas = z_meas - true_pitch;
err_ekf = x_history(:,1) - true_pitch;
plot(time, err_meas, 'g', 'DisplayName', 'Sensor Error'); hold on;
plot(time, err_ekf, 'b', 'LineWidth', 1.5, 'DisplayName', 'EKF Error');
yline(0, 'k--');
title('Estimation Error');
xlabel('Time (s)');
ylabel('Error (rad)');
legend('Location', 'best');
grid on;

% Calculate RMSE
rmse_meas = sqrt(mean(err_meas.^2));
rmse_ekf = sqrt(mean(err_ekf.^2));

fprintf('RMSE Sensor: %.4f\n', rmse_meas);
fprintf('RMSE EKF:    %.4f\n', rmse_ekf);
fprintf('Improvement: %.2f%%\n', (1 - rmse_ekf/rmse_meas)*100);