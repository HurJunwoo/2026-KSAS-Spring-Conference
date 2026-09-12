% 2026 KSAS Spring Conference

clear; clc; close all;
rng(2); 

%% 1. 시나리오 및 파라미터
dt = 0.01;
max_time = 100;
g = 9.81;

% Missile Initial Condition
Vm = 200;               
Pm = [0; 0; 0];         
theta0 = deg2rad(50); psi0 = deg2rad(0); 
em = [cos(theta0)*cos(psi0); cos(theta0)*sin(psi0); sin(theta0)]; 
am = [0;0;0];
tau = 0.1;

% Target Property
Vt = 50;               
Pt = [6000; 2000; 0]; 
et = [1; 0; 0];
at0 = [2*g; 5*g; 0];
aT_max_mc = 9.0 * g; 
alpha_T0 = atan(et(3)/norm(et(1:2)));
beta_T0 = atan(et(2)/et(1));

current_phase = 1;

% Parameters
sigma_max = deg2rad(15);  
aM_max = 20 * g;
aM_max_N = 0.9*aM_max;
aM_max_virtual = 0.5*aM_max;

if Vt >= Vm * sin(sigma_max)
    fprintf("Warning: Vm insufficient (Limit > %.2f)\n", Vt / sin(sigma_max));
end

% Guidance Gains
% N_initial = 4.0;   % Phase 3 Guidance Gain   
% N_max = 9.0;       % Phase 4 Guidance Gain
N_initial = 2*aM_max_N / (cos(sigma_max)*aM_max_N - aT_max_mc);
fprintf("PN Gain : %.5f\n", N_initial);
K_FOV = 2*N_initial;
sigma_margin = deg2rad(2); 

% HNG Parameter
HNG_N = 1/(cos(sigma_max-sigma_margin)*0.9);       % Helical Basic Guidance Gain
c_Z = 3.5;                            % Weaving Parameter
c_Y = 3.5;                            % Weaving Parameter
HNG_freq_factor = 2.0;                % Weaving Frequency

% FOV Control
FOV_relock = sigma_max;         
FOV_max_virtual = FOV_relock;

% EKF: Bearing-Only Estimation Property
sig_ang = deg2rad(0.5);   
R_meas = diag([sig_ang^2, sig_ang^2]); 

% Initial Range Error
initial_range_guess_error = -500; 
d_true_init = Pt - Pm;
r_true_init = norm(d_true_init);
u_vec_init = d_true_init / r_true_init;
Pt_guess = Pm + (r_true_init + initial_range_guess_error) * u_vec_init;

X_est = [Pt_guess; Vt*et; at0]; 
P_cov = diag([1000^2, 1000^2, 1000^2, 100, 100, 100, 1, 1, 1]); % Initial Uncertainty Covariance

F = [eye(3), eye(3)*dt, 0.5*eye(3)*dt^2; zeros(3), eye(3), eye(3)*dt; zeros(3), zeros(3), eye(3)];
q_jerk = 0.5; 
G = [0.5*dt^2*eye(3); dt*eye(3); eye(3)];
Q = G * q_jerk^2 * G';

Pt_filt = X_est(1:3); Vt_vec_filt = X_est(4:6); 
Niter = max_time / dt;

% History
history = struct('Pm', [], 'Pt', [], 'phase', [], 'sigma_m', [], ...
                 'FOV_limit', [], 'P_virt', [], 'acc_m', [], ...
                 'r_est', [], 'r_true', [], 'pos_unc', [], 'am_vec', []); 
logger.t = nan(1, Niter);
logger.FOV = nan(1, Niter); logger.PN = nan(1, Niter); logger.Helical = nan(1, Niter);
logger.w_hng = nan(1, Niter);

% Phase Variables
P_virtual = [];
P_acq_center = []; % Phase 2에서 바라볼 불확실성 중심점

T_HNG = pi/HNG_freq_factor; T_conv = 10*tau;
phase4_threshold = (Vm+Vt)*(T_HNG+T_conv)/0.9;  % 고정 거리가 아닌 시간 계산 매커니즘 추가
fprintf("Phase4 Threshold = %.2f m\n", phase4_threshold);

phase2_time = nan; phase3_time = nan; phase4_time = nan; hit_time = nan;
min_dist = inf;
terminate_sim = false;
success = false;
r_prev_step = norm(X_est(1:3) - Pm);

fprintf("Simulation Started (Ver: MC Centroid Aiming)\n");

%% Simulation Loop
for t_idx = 1:(max_time/dt)
    t = t_idx * dt;
    
    % Physics
    d_vec_true = Pt - Pm; r_true = norm(d_vec_true);
    eL_true = d_vec_true / r_true;
    sigma_m_true = acos(max(min(dot(em, eL_true), 1), -1));
    
    % EKF Update
    is_visible = (sigma_m_true <= sigma_max);
    
    X_pred = F * X_est; P_pred = F * P_cov * F' + Q;

    if t <= 5
        at = at0;                % 0 ~ 5초
    elseif t <= 10
        at = [3*g; 0; 5*g];      % 5 ~ 10초
    elseif t <= 20
        at = [5*g; g; 0];        % 10 ~ 20초
    elseif t <= 30
        at = [2*g; 5*g; -2*g];   % 20 ~ 30초
    else
        at = [5*g; -3*g; 0];      % 30초 이후
    end
    
    % at = at0;

    if is_visible
        z_meas = [atan2(d_vec_true(2), d_vec_true(1)) + randn*sig_ang; 
                  asin(d_vec_true(3)/r_true) + randn*sig_ang];
        
        dP_est = X_pred(1:3) - Pm; r_est = norm(dP_est);
        H = calculate_jacobian_bot(dP_est(1), dP_est(2), dP_est(3), r_est);
        
        z_pred = [atan2(dP_est(2), dP_est(1)); asin(dP_est(3)/r_est)];
        residual = z_meas - z_pred; 
        residual(1) = wrapToPi(residual(1)); residual(2) = wrapToPi(residual(2));
        
        S = H * P_pred * H' + R_meas;
        K = P_pred * H' / S;
        X_new = X_pred + K * residual; 
        P_new = (eye(9) - K * H) * P_pred;
        
        % [Projection Filter Logic]
        dP_new = X_new(1:3) - Pm; 
        r_new = norm(dP_new);
        max_delta_r = (Vm + Vt) * dt * 2.0; 
        if max_delta_r < 10, max_delta_r = 10; end
        
        % if abs(r_new - r_prev_step) > max_delta_r
        %     u_est_new = dP_new / (r_new + 1e-9);
        %     if r_new > r_prev_step + max_delta_r
        %         r_clamped = r_prev_step + max_delta_r;
        %     else
        %         r_clamped = r_prev_step - max_delta_r;
        %     end
        %     X_new(1:3) = Pm + r_clamped * u_est_new;
        % end
        
        X_est = X_new; 
        P_cov = P_new;
    else
        X_est = X_pred; P_cov = P_pred;
    end
    
    r_prev_step = norm(X_est(1:3) - Pm);
    
    % Guidance Variables
    alpha_use = 0.5;   % LPF Gain (1에 가까울수록 최신값 반영 비율 증가)
    Pt_filt = (1 - alpha_use) * Pt_filt + alpha_use * X_est(1:3);
    Vt_vec_filt = (1 - alpha_use) * Vt_vec_filt + alpha_use * X_est(4:6);
    
    Pt_G = Pt_filt; Vt_vec_G = Vt_vec_filt; 
    
    dP_G = Pt_G - Pm; r_est_G = norm(dP_G); el_est_G = dP_G / r_est_G;
    Vr_vec_est = Vm*em - Vt_vec_G; Vr_est = norm(Vr_vec_est);
    sigma_m_est = acos(max(min(dot(em, el_est_G), 1), -1));
    
    % Phase
    if current_phase == 1
        % Phase 1
        if isempty(P_virtual)
            % --- [Step 1] 초기화 및 수렴 파라미터 ---
            dist_init = norm(Pt_G - Pm);
            T_blind_est = (dist_init) / Vm;  % 최초 예상 시간
            max_iter = 10;                  % 수렴을 위해 횟수 증가
            tol = 1;                    % 수렴 오차 (초)
            % damping = 1;                 % 업데이트 감쇠 (진동 방지)
            
            fprintf(">> [Iteration] Starting Fixed-point Iteration for T_blind...\n");

            for iter = 1:max_iter
                T_old = T_blind_est;
                
                num_samples = 2000;      
                mc_phys_dt = dt;      
                mc_maneuver_dt = 1.0; 
                
                total_mc_steps = ceil(T_blind_est / mc_phys_dt);
                steps_per_maneuver = round(mc_maneuver_dt / mc_phys_dt);
                
                pos_sim = repmat(Pt_G, 1, num_samples);
                vel_sim = repmat(Vt * et, 1, num_samples); 
                acc_cmd = repmat(at0, 1, num_samples); 
                
                for k = 1:total_mc_steps
                    if mod(k-1, steps_per_maneuver) == 0
                        change_idx = rand(1, num_samples) < 0.10;
                        num_change = sum(change_idx);
                        if num_change > 0
                            rand_vec = randn(3, num_change);
                            v_dir = vel_sim(:, change_idx) ./ (sqrt(sum(vel_sim(:, change_idx).^2, 1)) + 1e-9);
                            dot_prod = sum(rand_vec .* v_dir, 1); 
                            ortho_vec = rand_vec - bsxfun(@times, v_dir, dot_prod); 
                            acc_dir = ortho_vec ./ (sqrt(sum(ortho_vec.^2, 1)) + 1e-9); 
                            acc_mag = (rand(1, num_change).^0.5) * aT_max_mc; 
                            acc_cmd(:, change_idx) = bsxfun(@times, acc_dir, acc_mag);
                        end
                    end
                    vel_sim = vel_sim + acc_cmd * mc_phys_dt;
                    v_new_mag = sqrt(sum(vel_sim.^2, 1));
                    vel_sim = bsxfun(@times, vel_sim ./ (v_new_mag + 1e-9), Vt); 
                    pos_sim = pos_sim + vel_sim * mc_phys_dt;
                end

                % --- [Step 3] 기하학적 목표점 및 고도 계산 ---
                P_mean_mc = mean(pos_sim, 2);
                error_vecs = pos_sim - P_mean_mc;
                dist_errors = sqrt(sum(error_vecs.^2, 1));
                target_deviation_prob = prctile(dist_errors, 99); 
                
                % 요구 고도 (안전 마진 포함)
                safety_margin = 1.0; 
                Required_Altitude = max(2000, (target_deviation_prob * safety_margin) / tan(sigma_max));
                % Required_Altitude = min(Required_Altitude, 6000); % 고도 상한선 제어
                
                % Waypoint 위치 선정 (Pull-back)
                Pt_wp_dir = P_mean_mc - Pt_G; 
                if norm(Pt_wp_dir) > 1e-3, Pt_wp_dir = Pt_wp_dir / norm(Pt_wp_dir); else, Pt_wp_dir = et; end
                
                % % 진입각 계산 및 클램핑 (Waypoint가 무한히 밀려나는 것 방지)
                % alpha_T = 2*(atan(Pt_wp_dir(3)/norm(Pt_wp_dir(1:2) + 1e-9)) - alpha_T0);
                % dive_angle_rad = asin(max(min((Vm/Vt)*sin(sigma_max), 1), -1)) - alpha_T;
                % pullback_dist = Required_Altitude / tan(max(dive_angle_rad, deg2rad(10))); % 최소 10도 보장
                
                vec_to_m = (Pm(1:2) - P_mean_mc(1:2));
                max_pullback = norm(vec_to_m); 
                % pullback_dist = min(pullback_dist, max_pullback);
                pullback_dist = max_pullback;
                
                dir_xy = [vec_to_m / (norm(vec_to_m) + 1e-9); 0]; 
                P_virtual_tmp = P_mean_mc + dir_xy * pullback_dist + [0; 0; Required_Altitude];
                
                % --- [Step 4] 시간 업데이트 및 수렴 판정 ---
                T_new = norm(P_virtual_tmp - Pm) / Vm;
                
                if abs(T_new - T_old) < tol || iter == max_iter
                    % 최종 값 확정
                    P_virtual = P_virtual_tmp;
                    P_acq_center = P_mean_mc;
                    
                    P_straight = Pt_G; et0 = et;
                    for m = 1:round(T_blind_est/dt)
                        et0 = et0 + (at0/Vt)*dt; et0 = et0/norm(et0);
                        P_straight = P_straight + Vt*et0*dt;
                    end

                    global prob_debug;
                    prob_debug.P_start = Pt_G; 
                    prob_debug.Vt = Vt;
                    prob_debug.pos_cloud = pos_sim; 
                    prob_debug.P_mean = P_mean_mc;
                    prob_debug.deviation = target_deviation_prob;
                    prob_debug.P_straight = P_straight;
                    
                    fprintf(">> [MC Converged] Iter: %d | T_blind: %.2f s | Radius: %.1f m | Loft: %.1f m\n", ...
                            iter, T_blind_est, target_deviation_prob, Required_Altitude);
                    break; 
                end
            end
        end
        

        % 실시간 기하학적 조건 검사 (FOV 내에 확률 구름이 들어오는지 확인)
        dist_to_centroid = norm(P_acq_center - Pm);
        
        % 미사일 현재 위치에서 구름 전체를 덮기 위해 필요한 시야각
        current_req_fov = atan(target_deviation_prob / max(1, dist_to_centroid)); 
        
        % 종료 조건 1: 기하학적으로 FOV 안에 구름이 여유 있게 들어옴
        cond_fov_ok = (current_req_fov < (sigma_max - sigma_margin));
        
        % 종료 조건 2: 요구 고도에 도달함 (Z축 오차 500m 이내)
        cond_alt_reached = (Pm(3) >= P_virtual(3) - 500);

        if cond_fov_ok && cond_alt_reached
            current_phase = 2; phase2_time = t;
            fprintf(">> Phase 2 (Acq) t=%.2f. Cloud FOV Req: %.1f deg (Limit: %.1f deg) | Alt: %.1f m\n", ...
                t, rad2deg(current_req_fov), rad2deg(sigma_max), Pm(3));
        end
        
        u_des = (P_virtual - Pm) / norm(P_virtual - Pm);
        an = 4.0 * Vm * cross(cross(em, u_des)/dt, em); 
        
    elseif current_phase == 2
        % Phase 2
        P_aim = P_acq_center; 
        
        vec_to_aim = P_aim - Pm;
        u_des = vec_to_aim / (norm(vec_to_aim) + 1e-9);
        
        % Vector Pursuit Guidance
        Kp_turn = 15.0; 
        err_vec = u_des - em;
        an = Kp_turn * Vm * err_vec;
        
        % Phase 3 전환 조건
        pos_uncertainty = norm(sqrt(diag(P_cov(1:3,1:3))));
        
        % 조건 1: 시야에 들어옴
        % 조건 2: 불확실성이 어느 정도 감소함
        if is_visible && pos_uncertainty < 500
            current_phase = 3; phase3_time = t;
            
            X_est(4:6) = [0; 0; 0]; 
            P_cov = diag([200^2, 200^2, 200^2, 300^2, 300^2, 300^2, 1^2, 1^2, 1^2]);     
            fprintf(">> Phase 3 START: Lock-on Confirmed (Unc: %.1f m)\n", pos_uncertainty);
        end
        
    elseif current_phase == 3 || current_phase == 4
        % Phase 3 & 4 (PN Guidance)
        if current_phase == 3 && r_est_G < phase4_threshold
            current_phase = 4; phase4_time = t;
            fprintf(">> Phase 4 (Hard Terminal) t=%.2f\n", t);
        end
        

        % q_dynamic = 3.0 * exp(-(t - phase3_time)/2.0) + 1.0; % 2초 동안 3.0에서 1.0으로 감쇠
        % Q = G * q_dynamic^2 * G';   
        Omega_rate = cross(Vr_vec_est, el_est_G) / r_est_G;
        
        % if current_phase == 4
        %     gain_factor = max(0, min(1, (1500 - r_est_G) / 1500));
        %     gain_factor = 0;
        %     N_use = N_initial + (N_max - N_initial) * gain_factor^2;
        % else
        %     N_use = N_initial;
        % end
        N_use = N_initial;
        
        an_PN = N_use * Vm * cross(Omega_rate, em);
        
        % HNG Bias
        an_HNG_Bias = [0;0;0];
        if current_phase == 3
            raw_HNG = calculate_HNG_accel(Vm, em, el_est_G, r_est_G, t, HNG_N, c_Z, c_Y, HNG_freq_factor);
            an_HNG_Bias = raw_HNG;
        end
        
        % FOV Constraint
        sigma_start = FOV_max_virtual - sigma_margin;
        w = 0;
        if sigma_m_est >= sigma_start
            w = (sigma_m_est - sigma_start) / (FOV_max_virtual - sigma_start);
            w = w^2 * (2 - w);
        end
        w = max(0, min(w, 1));
        if current_phase == 4, w = 0; end 
        
        omega_L_vec = cross(el_est_G, Vr_vec_est) / r_est_G;
        dot_sigma_natural = -dot(omega_L_vec, cross(em, el_est_G)/norm(cross(em, el_est_G) + 1e-9));
        perp_vec = el_est_G - dot(el_est_G, em) * em;
        u_fov = perp_vec / (norm(perp_vec) + 1e-9);
        am_FOV = (Vm * dot_sigma_natural) * u_fov + (K_FOV * Vm * (sigma_m_est - sigma_start)) * u_fov;
        
        logger.FOV(t_idx) = norm(w*am_FOV);
        logger.PN(t_idx) = norm((1-w)*an_PN);
        logger.Helical(t_idx) = norm((1-w)*an_HNG_Bias);

        % % t_idx 루프 내부 (Phase 3 상황)
        % if current_phase == 3
        %     r_high = phase4_threshold*1.5;      % 위빙을 줄이기 시작하는 거리
        %     r_low  = phase4_threshold; % 위빙을 완전히 끌 거리 (0.5는 예시)
        % 
        %     if r_est_G >= r_high
        %         w_hng = 1;
        %     elseif r_est_G <= r_low
        %         w_hng = 0;
        %     else
        %         % r_low ~ r_high 구간에서 0과 1 사이를 부드럽게 연결 (S-curve)
        %         ratio = (r_est_G - r_low) / (r_high - r_low);
        %         w_hng = 0.5 * (1 - cos(pi * ratio)); 
        %     end
        % else
        %     w_hng = 0;
        % end
        % 
        % logger.w_hng(t_idx) = w_hng;

        an_Base = an_PN + an_HNG_Bias;
        an = (1 - w) * an_Base + w * am_FOV;
    end
    
    if norm(an) > aM_max, an = an/norm(an)*aM_max; end
    an = an - dot(an, em)*em; 

    am = am + ((an - am) / tau) * dt; 
    am = am - dot(am, em)*em;

    % am = an; 

    % Simulation Loop 내부의 History Update 부분 수정
    history.am_vec(:, end+1) = am; % 3D 벡터 통째로 저장
    
    % Physics Update
    Pm = Pm + Vm*em*dt; em = em + (am/Vm)*dt; em = em/norm(em); 
    et = et + (at/Vt)*dt; et = et/norm(et);
    Pt = Pt + Vt*et*dt;
    
    % History Update
    current_pos_uncertainty = norm(sqrt(diag(P_cov(1:3,1:3))));
    history.pos_unc(end+1) = current_pos_uncertainty;
    
    history.Pm(:,end+1) = Pm; history.Pt(:,end+1) = Pt; 
    history.phase(end+1) = current_phase;
    history.sigma_m(end+1) = rad2deg(sigma_m_true);
    history.FOV_limit(end+1) = rad2deg(FOV_max_virtual);
    if isempty(P_virtual), history.P_virt(:,end+1) = [nan;nan;nan]; else, history.P_virt(:,end+1) = P_virtual; end
    history.acc_m(end+1) = norm(am);
    history.r_true(end+1) = r_true;
    history.r_est(end+1) = r_est_G;
    logger.t(t_idx) = t;
    
    min_dist = min(min_dist, r_true);
    if r_true < 5.0 || terminate_sim
        V_rel_vec = (Vm*em) - (Vt*et); exact_miss = norm(cross(d_vec_true, V_rel_vec)) / norm(V_rel_vec);
        ang_final = rad2deg(acos(max(min(dot(em, et), 1), -1)));
        min_dist = min(min_dist, exact_miss);
        if exact_miss <= 2.0, status_res="HIT"; else, status_res="MISS"; end
        fprintf("\n=========== RESULT ==============\n Status: %s\n Miss: %.2f m | Impact Angle: %.1f deg\n", status_res, exact_miss, ang_final);
        hit_time = t;
        break;
    end
end
if ~success && ~terminate_sim && t_idx == Niter, fprintf("Failed: Time Limit. Min Dist: %.2f\n", min_dist); end

%% Plots
% 1. Monte Carlo Cloud
if exist('prob_debug', 'var') && isfield(prob_debug, 'pos_cloud')
    figure('Theme','light','Color','w', 'Name', 'Target Probability Cloud');
    hold on; grid on; axis equal; view(3);
    plot3(history.Pm(1,:), history.Pm(2,:), history.Pm(3,:), 'b-', 'LineWidth', 1.2, 'DisplayName', 'Missile');
    plot3(history.Pt(1,:), history.Pt(2,:), history.Pt(3,:), 'k--', 'LineWidth', 1.2, 'DisplayName', 'Target');
    
    cloud = prob_debug.pos_cloud;
    idx_draw = randperm(size(cloud,2), min(500, size(cloud,2)));
    scatter3(cloud(1,idx_draw), cloud(2,idx_draw), cloud(3,idx_draw), 10, 'r', 'filled', 'MarkerFaceAlpha', 0.3, 'DisplayName', 'Monte-Carlo Point');
    
    c = prob_debug.P_mean; r_99 = prob_debug.deviation;
    [x,y,z] = sphere(20);
    surf(r_99*x+c(1), r_99*y+c(2), r_99*z+c(3), 'FaceColor', 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', 'Monte-Carlo Sphere');
    
    wp = history.P_virt(:, find(~isnan(history.P_virt(1,:)), 1));
    plot3(wp(1), wp(2), wp(3), 'mx', 'MarkerSize', 12, 'LineWidth', 3, 'DisplayName', 'Phase 1 Waypoint');
    line([wp(1), c(1)], [wp(2), c(2)], [wp(3), c(3)], 'Color', 'm', 'LineStyle', ':', 'DisplayName', 'Vertical line');
    plot3(c(1), c(2), c(3), 'gx', 'MarkerSize', 12, 'LineWidth', 3, 'DisplayName', 'Centroid');
    
    title('Target Probability');
    legend('Location', 'best');
end

% 2. Trajectory
figure('Theme','light','Color','w', 'Name', 'Trajectory'); 
hold on; grid on; axis equal; view(3);
plot3(history.Pt(1,:),history.Pt(2,:),history.Pt(3,:),'k--','LineWidth',1.2,'DisplayName','Target Path');
idx1=history.phase==1; idx2=history.phase==2; idx3=history.phase==3; idx4=history.phase==4;
if any(idx1), plot3(history.Pm(1,idx1),history.Pm(2,idx1),history.Pm(3,idx1),'b-','LineWidth',1.5,'DisplayName','Phase 1'); end
if any(idx2), plot3(history.Pm(1,idx2),history.Pm(2,idx2),history.Pm(3,idx2),'c-','LineWidth',1.5,'DisplayName','Phase 2'); end
if any(idx3), plot3(history.Pm(1,idx3),history.Pm(2,idx3),history.Pm(3,idx3),'g-','LineWidth',1.5,'DisplayName','Phase 3'); end
if any(idx4), plot3(history.Pm(1,idx4),history.Pm(2,idx4),history.Pm(3,idx4),'r-','LineWidth',2.5,'DisplayName','Phase 4'); end
if ~all(isnan(history.P_virt(:,1))), plot3(history.P_virt(1,1), history.P_virt(2,1), history.P_virt(3,1), 'mx', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'WayPoint'); end
legend('Location','best'); title('Trajectory'); xlabel('X'); ylabel('Y'); zlabel('Z');

% 3. Position Uncertainty
figure('Theme','light','Color','w', 'Name', 'Position Uncertainty Analysis');
grid on; hold on;
plot(logger.t(1:length(history.pos_unc)), history.pos_unc, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Pos Uncertainty (Norm)');
yline(500, 'r--', 'LineWidth', 2, 'DisplayName', 'Threshold (1000m)');
if ~isnan(phase2_time), xline(phase2_time, 'c:', 'LineWidth', 1.5, 'DisplayName', 'Phase2 start'); end
if ~isnan(phase3_time), xline(phase3_time, 'g:', 'LineWidth', 1.5, 'DisplayName', 'Phase3 start'); end
if ~isnan(phase4_time), xline(phase4_time, 'm:', 'LineWidth', 1.5, 'DisplayName', 'Phase4 start'); end
xlabel('Time (s)'); ylabel('Uncertainty (m)'); title('EKF Position Uncertainty'); legend('Location', 'best');

% 4. FOV
figure('Theme','light','Color','w', 'Name', 'FOV Analysis'); 
grid on; hold on;
yline(rad2deg(sigma_max), 'r--', 'LineWidth', 2, 'DisplayName', 'Physical Limit');
plot(logger.t(1:length(history.sigma_m)), history.sigma_m, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Real Look Angle');
if ~isnan(phase2_time), xline(phase2_time, 'c:', 'LineWidth', 1.5, 'DisplayName', 'Phase2 start'); end
if ~isnan(phase3_time), xline(phase3_time, 'r:', 'LineWidth', 1.5, 'DisplayName', 'Phase3 start'); end
if ~isnan(phase4_time), xline(phase4_time, 'm:', 'LineWidth', 1.5, 'DisplayName', 'Phase4 start'); end
xlabel('Time (s)'); ylabel('Look Angle (deg)'); title('Seeker FOV'); legend('Location', 'North');

% 5. Range Estimation
figure('Theme','light','Color','w', 'Name', 'Range Estimation');
grid on; hold on;

len_data = length(history.r_true);
time_axis = logger.t(1:len_data);

plot(time_axis, history.r_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True Range');
plot(time_axis, history.r_est, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Estimated Range');
if ~isnan(phase2_time), xline(phase2_time, 'c:', 'LineWidth', 1.5, 'DisplayName', 'Phase2 start'); end
if ~isnan(phase3_time), xline(phase3_time, 'g:', 'LineWidth', 1.5, 'DisplayName', 'Phase3 start'); end
if ~isnan(phase4_time), xline(phase4_time, 'm:', 'LineWidth', 1.5, 'DisplayName', 'Phase4 start'); end
if ~isnan(hit_time), xline(hit_time, 'k:', 'LineWidth', 1.5, 'DisplayName', 'Hit time'); end
xlabel('Time (s)'); ylabel('Range (m)'); 
title('Range Estimation'); 
legend('Location', 'best');

% 6. Acceleration
figure('Theme','light','Color','w', 'Name', 'Acceleration Profile');
grid on; hold on;

len_data = length(history.acc_m);
time_axis = logger.t(1:len_data);
acc_g = history.acc_m / g;

plot(time_axis, acc_g, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Cmd Acceleration');
yline(aM_max/g, 'r--', 'LineWidth', 2, 'DisplayName', 'Limit (20g)');
if ~isnan(phase2_time), xline(phase2_time, 'c:', 'LineWidth', 1.5, 'DisplayName', 'Phase2 start'); end
if ~isnan(phase3_time), xline(phase3_time, 'g:', 'LineWidth', 1.5, 'DisplayName', 'Phase3 start'); end
if ~isnan(phase4_time), xline(phase4_time, 'm:', 'LineWidth', 1.5, 'DisplayName', 'Phase4 start'); end
if ~isnan(hit_time), xline(hit_time, 'k:', 'LineWidth', 1.5, 'DisplayName', 'Hit time'); end
xlabel('Time (s)'); 
ylabel('Acceleration (g)'); 
title('Acceleration'); 
legend('Location', 'best');

% 7. Phase 3 Acceleration
figure('Theme','light','Color','w', 'Name', 'Phase 3 Acceleration');
grid on; hold on;
len_data = length(logger.FOV);
time_axis = logger.t(1:len_data);
plot(time_axis, logger.FOV, 'b', 'LineWidth', 1.5, 'DisplayName', 'FOV Acceleration');
plot(time_axis, logger.PN, 'r', 'LineWidth', 1.5, 'DisplayName', 'PN Acceleration');
plot(time_axis, logger.Helical, 'k', 'LineWidth', 1.5, 'DisplayName', 'Helical Acceleration');
if ~isnan(phase2_time), xline(phase2_time, 'c:', 'LineWidth', 1.5, 'DisplayName', 'Phase2 start'); end
if ~isnan(phase3_time), xline(phase3_time, 'g:', 'LineWidth', 1.5, 'DisplayName', 'Phase3 start'); end
if ~isnan(phase4_time), xline(phase4_time, 'm:', 'LineWidth', 1.5, 'DisplayName', 'Phase4 start'); end
if ~isnan(hit_time), xline(hit_time, 'k:', 'LineWidth', 1.5, 'DisplayName', 'Hit time'); end
xlabel('Time (s)'); 
ylabel('Acceleration'); 
title('Phase 3 Acceleration'); 
legend('Location', 'best');

% 8. Helical Guidance
% Phase 3 데이터만 추출
idx3 = (history.phase == 3);
ay = history.am_vec(2, idx3);
az = history.am_vec(3, idx3);
t_phase3 = logger.t(idx3);

figure('Color', 'w', 'Theme', 'light', 'Name', 'Helical Acceleration');
plot(ay, az, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Accel Trajectory');
hold on; grid on; axis equal;

% FOV가 개입된 시점(Blue line이 솟구친 시점)만 강조
% (FOV 가속도 크기가 일정 수준 이상인 곳을 찾음)
% 이 예시를 위해 history에 fov_acc 성분이 저장되어 있다고 가정
% scatter(ay(fov_active), az(fov_active), 10, 'r', 'filled'); 

xlabel('a_y (m/s^2)'); ylabel('a_z (m/s^2)');
title('Helical Guidance');

% % 9. Weight of HNG
% figure('Color', 'w', 'Theme', 'light', 'Name', 'Weight of HNG');
% plot(logger.t, logger.w_hng, 'b', 'LineWidth', 1.5, 'DisplayName', 'HNG Weight');
% if ~isnan(phase2_time), xline(phase2_time, 'c:', 'LineWidth', 1.5, 'DisplayName', 'Phase2 start'); end
% if ~isnan(phase3_time), xline(phase3_time, 'g:', 'LineWidth', 1.5, 'DisplayName', 'Phase3 start'); end
% if ~isnan(phase4_time), xline(phase4_time, 'm:', 'LineWidth', 1.5, 'DisplayName', 'Phase4 start'); end
% hold on; grid on;
% legend('Location', 'best');
% title('Weight of HNG');
% xlabel('Time (s)'); ylabel('Weight');
% ylim([0, 3]);
%% Functions
function v_w = wrapToPi(v), v_w = mod(v+pi,2*pi)-pi; end
function H = calculate_jacobian_bot(dx, dy, dz, r)
    rho2 = dx^2 + dy^2; rho = sqrt(rho2);
    if rho < 1e-6, rho = 1e-6; rho2 = rho^2; end
    if r < 1e-6, r = 1e-6; end
    r2 = r^2;
    dAz_dx = -dy / rho2; dAz_dy = dx / rho2; dAz_dz = 0;
    dEl_dx = -dx * dz / (r2 * rho); dEl_dy = -dy * dz / (r2 * rho); dEl_dz = rho / r2;
    H = [dAz_dx, dAz_dy, dAz_dz, 0, 0, 0, 0, 0, 0; dEl_dx, dEl_dy, dEl_dz, 0, 0, 0, 0, 0, 0];
end
function a_HN = calculate_HNG_accel(Vm, v_vec, lambda, r_hat, t, N, c_Z, c_Y, HNG_freq_factor)
    e_X = [1; 0; 0]; 
    eta_vec = cross(e_X, lambda);
    if norm(eta_vec) < 1e-6, eta_vec = [0;1;0]; end 
    eta_vec = eta_vec / norm(eta_vec);
    theta_eta = acos(max(min(dot(lambda, e_X), 1), -1));
    K = [0 -eta_vec(3) eta_vec(2); eta_vec(3) 0 -eta_vec(1); -eta_vec(2) eta_vec(1) 0];
    R_eta = eye(3) + sin(theta_eta)*K + (1-cos(theta_eta))*(K^2);
    r_metric = r_hat / 500; 
    if r_metric < 0.5, r_metric = 0.5; end 
    if r_metric > 5, r_metric = 5; end
    angle_Z = c_Z * sin(HNG_freq_factor * t) / r_metric; 
    angle_Y = c_Y * cos(HNG_freq_factor * t) / r_metric;
    R_Z = [cos(angle_Z) -sin(angle_Z) 0; sin(angle_Z) cos(angle_Z) 0; 0 0 1];
    R_Y = [cos(angle_Y) 0 sin(angle_Y); 0 1 0; -sin(angle_Y) 0 cos(angle_Y)];
    R_total = R_eta * R_Z * R_Y;
    rotated_vec = R_total * e_X;
    v_full = Vm * v_vec;
    vec_term = cross(rotated_vec, lambda);
    a_HN = N * cross(v_full, vec_term);
end
