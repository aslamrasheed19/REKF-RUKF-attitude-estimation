%% Robust Kalman Filter Simulation (Euler Integration for Truth & Filter)
% This script ensures Model Match so S=1 before/after the fault.

clear; clc; close all;

%% 1. Simulation Parameters
% -------------------------------------------------------------------------
% Constants
mu = 3.98601e14;            % Earth gravitational constant
Re = 6378137;               % Earth radius
h_alt = 550000;             % Altitude
r0 = Re + h_alt;            % Orbit radius
w0 = sqrt(mu / r0^3);       % Orbit angular velocity
inc = deg2rad(97);          % Inclination
epsilon = deg2rad(11.7);    % Magnetic dipole tilt
we = 7.29e-5;               % Earth spin rate
Me = 7.943e15;              % Magnetic dipole moment

% Simulation Timing
T_sim = 500;                % Total simulation time (s)
dt = 0.1;                   % Sampling time (s)
time = 0:dt:T_sim;
N = length(time);

% Satellite Inertia [cite: 276-282]
Jx = 2.1e-3; Jy = 2.0e-3; Jz = 1.9e-3;
J = diag([Jx, Jy, Jz]);
J_inv = inv(J);

% Filter Tuning
sigma_m = 200e-9;           % Assumed Magnetometer noise (Tesla)
R_cov = (sigma_m^2) * eye(3); 
% Small process noise to allow filter to work, but model is perfect
Q_cov = diag([1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10]); 

% Robust Threshold
chi2_threshold = 12.592;    

%% 2. True Trajectory Generation (EULER METHOD)
% -------------------------------------------------------------------------
fprintf('Generating True Trajectory (Euler Integration)...\n');

% Initial State: [phi, theta, psi, wx, wy, wz]
x_true = zeros(6, N);
x_true(:,1) = [deg2rad(10); deg2rad(10); deg2rad(10); 0.001; 0.001; 0.001];

for k = 1:N-1
    x_curr = x_true(:,k);
    
    % --- EULER INTEGRATION STEP ---
    dx = satellite_dynamics(x_curr, J, J_inv, w0, mu, r0);
    x_true(:,k+1) = x_curr + dt * dx; 
end

%% 3. Measurement Generation & Fault Injection
% -------------------------------------------------------------------------
y_meas = zeros(3, N);

for k = 1:N
    t = time(k);
    
    % True Measurement (Perfect Model)
    phi=x_true(1,k); theta=x_true(2,k); psi=x_true(3,k);
    Aa = euler2dcm(phi, theta, psi);
    H_orb = mag_field_model(t, r0, w0, inc, epsilon, we, Me);
    H_body = Aa * H_orb;
    
    % NO NOISE ADDED (Perfect Sensors) to demonstrate S=1
    y_meas(:,k) = H_body;
    
    % FAULT INJECTION (300s to 301s)
    if t >= 300 && t <= 301
        y_meas(1,k) = y_meas(1,k) + 5e-5; % Add 50,000 nT bias
    end
end

%% 4. Filter Initialization
% -------------------------------------------------------------------------
x_init = x_true(:,1); % Perfect initialization to show pure tracking
P_init = eye(6) * 1e-6;

% Storage
x_ekf = zeros(6, N); x_ekf(:,1) = x_init; P_ekf = P_init;
x_rekf = zeros(6, N); x_rekf(:,1) = x_init; P_rekf = P_init;
mnsf_rekf = ones(1, N);

x_ukf = zeros(6, N); x_ukf(:,1) = x_init; P_ukf = P_init;
x_rukf = zeros(6, N); x_rukf(:,1) = x_init; P_rukf = P_init;
mnsf_rukf = ones(1, N);

%% 5. Filter Loops
% -------------------------------------------------------------------------
fprintf('Running Filters...\n');

for k = 1:N-1
    t_next = time(k+1);
    y_next = y_meas(:, k+1);
    H_orb_next = mag_field_model(t_next, r0, w0, inc, epsilon, we, Me);
    
    % --- EKF & REKF ---
    % Both use Euler prediction inside 'ekf_predict'
    [x_ekf(:,k+1), P_ekf] = ekf_step(x_ekf(:,k), P_ekf, y_next, dt, J, J_inv, w0, mu, r0, H_orb_next, Q_cov, R_cov, 0, 0);
    [x_rekf(:,k+1), P_rekf, s_r] = ekf_step(x_rekf(:,k), P_rekf, y_next, dt, J, J_inv, w0, mu, r0, H_orb_next, Q_cov, R_cov, 1, chi2_threshold);
    mnsf_rekf(k+1) = s_r;
    
    % --- UKF & RUKF ---
    % Both use Euler prediction inside 'ukf_step'
    [x_ukf(:,k+1), P_ukf] = ukf_step(x_ukf(:,k), P_ukf, y_next, dt, J, J_inv, w0, mu, r0, H_orb_next, Q_cov, R_cov, 0, 0);
    [x_rukf(:,k+1), P_rukf, s_u] = ukf_step(x_rukf(:,k), P_rukf, y_next, dt, J, J_inv, w0, mu, r0, H_orb_next, Q_cov, R_cov, 1, chi2_threshold);
    mnsf_rukf(k+1) = s_u;
end

%% 6. Plotting Results
% -------------------------------------------------------------------------
rad2deg_c = 180/pi;

% Figure 1: EKF vs REKF
figure('Color','w');
subplot(2,1,1);
plot(time, x_true(2,:)*rad2deg_c, 'k', 'LineWidth', 1.5); hold on;
plot(time, x_ekf(2,:)*rad2deg_c, 'b--');
plot(time, x_rekf(2,:)*rad2deg_c, 'r-.');
legend('True', 'EKF', 'REKF');
title('Pitch Angle: EKF vs REKF (Euler Model Match)'); ylabel('Angle (deg)'); grid on;
%xlim([290 310]); % Zoom on fault

subplot(2,1,2);
plot(time, mnsf_rekf, 'm', 'LineWidth', 1.5);
title('REKF Measurement Noise Scale Factor (S)');
xlabel('Time (s)'); ylabel('S(k)'); grid on;
% Expect S=1 everywhere except 300-301s
%ylim([0 max(mnsf_rekf)*1.1]); 

% Figure 2: UKF vs RUKF
figure('Color','w');
subplot(2,1,1);
plot(time, x_true(2,:)*rad2deg_c, 'k', 'LineWidth', 1.5); hold on;
plot(time, x_ukf(2,:)*rad2deg_c, 'b--');
plot(time, x_rukf(2,:)*rad2deg_c, 'g-.');
legend('True', 'UKF', 'RUKF');
title('Pitch Angle: UKF vs RUKF (Euler Model Match)'); ylabel('Angle (deg)'); grid on;
%xlim([290 310]);

subplot(2,1,2);
plot(time, mnsf_rukf, 'm', 'LineWidth', 1.5);
title('RUKF Measurement Noise Scale Factor (S)');
xlabel('Time (s)'); ylabel('S(k)'); grid on;

% Figure 3: REKF vs RUKF Comparison
figure('Color','w');
plot(time, x_true(2,:)*rad2deg_c, 'k', 'LineWidth', 2); hold on;
plot(time, x_rekf(2,:)*rad2deg_c, 'g--', 'LineWidth', 1);
plot(time, x_rukf(2,:)*rad2deg_c, 'r-', 'LineWidth', 1);
legend('True', 'RUKF', 'REKF');
title('Comparison of Robust Filters (Zoomed)');
ylabel('Pitch (deg)'); xlabel('Time (s)');
%xlim([295 305]);
 grid on;

%% 7. Helper Functions
% -------------------------------------------------------------------------

function dx = satellite_dynamics(x, J, J_inv, w0, mu, r0)
    phi = x(1); theta = x(2); psi = x(3);
    w_bi = x(4:6); 
    
    c_th = cos(theta); s_th = sin(theta);
    c_ph = cos(phi);   s_ph = sin(phi);
    
    % Gravity Gradient
    A13 = -s_th; A23 = s_ph * c_th; A33 = c_ph * c_th;
    coef = -3 * mu / r0^3;
    Ngg = coef * [ (J(2,2)-J(3,3))*A23*A33;
                   (J(3,3)-J(1,1))*A13*A33;
                   (J(1,1)-J(2,2))*A13*A23 ];
               
    dw_bi = J_inv * (Ngg - cross(w_bi, J * w_bi));
    
    % Kinematics (Eq 5 approx)
    A = euler2dcm(phi, theta, psi);
    w_orbit = [0; -w0; 0];
    w_br = w_bi + A * w_orbit;
    
    tt = tan(theta); ct = cos(theta);
    H_mat = [1, sin(phi)*tt, cos(phi)*tt;
             0, cos(phi),    -sin(phi);
             0, sin(phi)/ct, cos(phi)/ct];
    dTheta = H_mat * w_br;
    
    dx = [dTheta; dw_bi];
end

function [x_new, P_new, S_out] = ekf_step(x, P, y, dt, J, J_inv, w0, mu, r0, H_orb, Q, R, robust, thresh)
    % Prediction (Euler)
    dx = satellite_dynamics(x, J, J_inv, w0, mu, r0);
    x_pred = x + dt * dx;
    
    % Jacobian F (Numerical)
    nx = 6; F = eye(nx); eps_j = 1e-6;
    for i=1:nx
        x_p = x; x_p(i) = x_p(i) + eps_j;
        dx_p = satellite_dynamics(x_p, J, J_inv, w0, mu, r0);
        x_pred_p = x_p + dt * dx_p;
        F(:,i) = (x_pred_p - x_pred)/eps_j;
    end
    P_pred = F * P * F' + Q;
    
    % Measurement
    h_val = euler2dcm(x_pred(1), x_pred(2), x_pred(3)) * H_orb;
    
    % Jacobian H (Numerical)
    H_jac = zeros(3, nx);
    for i=1:3
        x_p = x_pred; x_p(i) = x_p(i) + eps_j;
        h_p = euler2dcm(x_p(1), x_p(2), x_p(3)) * H_orb;
        H_jac(:,i) = (h_p - h_val)/eps_j;
    end
    
    innov = y - h_val;
    
    % Robust S
    S_out = 1;
    P_zz = H_jac * P_pred * H_jac';
    if robust
        beta = innov' * inv(P_zz + R) * innov;
        if beta > thresh
            S_out = (innov'*innov - trace(P_zz)) / trace(R);
            S_out = max(1, S_out);
        end
    end
    
    S_total = P_zz + S_out * R;
    K = P_pred * H_jac' / S_total;
    x_new = x_pred + K * innov;
    P_new = (eye(nx) - K * H_jac) * P_pred;
end

function [x_new, P_new, S_out] = ukf_step(x, P, y, dt, J, J_inv, w0, mu, r0, H_orb, Q, R, robust, thresh)
    n = 6; lambda = 3-n;
    % Sigma Points
    S_mat = chol((n+lambda)*(P+Q))';
    X_sig = [x, x+S_mat, x-S_mat];
    W_m = [lambda/(n+lambda), 0.5/(n+lambda)*ones(1,2*n)];
    W_c = W_m;
    
    % Prediction (Euler for each sigma point)
    X_pred = zeros(size(X_sig));
    for i=1:2*n+1
        dx = satellite_dynamics(X_sig(:,i), J, J_inv, w0, mu, r0);
        X_pred(:,i) = X_sig(:,i) + dt * dx;
    end
    
    x_pred = sum(X_pred .* W_m, 2);
    P_pred = Q;
    for i=1:2*n+1, d=X_pred(:,i)-x_pred; P_pred=P_pred+W_c(i)*(d*d'); end
    
    % Measurement
    Y_pred = zeros(3, 2*n+1);
    for i=1:2*n+1
        Y_pred(:,i) = euler2dcm(X_pred(1,i), X_pred(2,i), X_pred(3,i)) * H_orb;
    end
    y_mean = sum(Y_pred .* W_m, 2);
    
    P_yy = zeros(3,3); P_xy = zeros(6,3);
    for i=1:2*n+1
        yd = Y_pred(:,i)-y_mean; xd = X_pred(:,i)-x_pred;
        P_yy = P_yy + W_c(i)*(yd*yd');
        P_xy = P_xy + W_c(i)*(xd*yd');
    end
    
    innov = y - y_mean;
    
    S_out = 1;
    if robust
        beta = innov' * inv(P_yy + R) * innov;
        if beta > thresh
            S_out = (innov'*innov - trace(P_yy)) / trace(R);
            S_out = max(1, S_out);
        end
    end
    
    P_vv = P_yy + S_out*R;
    K = P_xy / P_vv;
    x_new = x_pred + K * innov;
    P_new = P_pred - K * P_vv * K';
end

function H = mag_field_model(t, r0, w0, inc, eps, we, Me)
    so = sin(w0*t); co = cos(w0*t); se = sin(eps); ce = cos(eps);
    si = sin(inc); ci = cos(inc); swe = sin(we*t); cwe = cos(we*t);
    t1 = ce*si - se*ci*cwe;
    H = (Me/r0^3) * [ co*t1 - so*se*swe; -(ce*ci + se*si*cwe); 2*(so*t1 + 2*co*se*swe) ];
end

function A = euler2dcm(ph, th, ps)
    cph=cos(ph); sph=sin(ph); cth=cos(th); sth=sin(th); cps=cos(ps); sps=sin(ps);
    A = [ cth*cps, cth*sps, -sth;
         -cph*sps+sph*sth*cps, cph*cps+sph*sth*sps, sph*cth;
          sph*sps+cph*sth*cps, -sph*cps+cph*sth*sps, cph*cth ];
end