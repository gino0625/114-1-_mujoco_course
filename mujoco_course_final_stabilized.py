import mujoco
import mujoco.viewer
import time
import numpy as np
import matplotlib.pyplot as plt # [新增] 引入繪圖庫
from numpy.linalg import inv

# --- 1. 物理參數設定 (基於 test.xml 結構估算) ---
g = 9.8

# M: 機器人本體總質量 (不含輪子)
# 參考 test.xml 馬達極限 (12Nm)，估算為中型機器人
M = 6.0     

# m: "單顆" 輪子質量
m = 0.4     

# l: 重心高度 (預估值)
l = 0.3     

# r: 輪子半徑 (預估值)
r = 0.08    

# D: 輪距 (左右輪中心距離)
D = 0.4   

# --- 2. 慣量計算 ---
# I: 單輪轉動慣量
I = 0.5 * m * r**2         

# Jd: 車身 Yaw 軸轉動慣量
Jd = (1/12) * M * (D**2)   

# Jp: 車身 Pitch 軸轉動慣量
Jp = (1/3) * M * (l**2)    

# --- 3. LQR 權重設定 ---
# [x, x_dot, theta, theta_dot, delta, delta_dot]
# 這裡保留對 delta (Yaw) 的控制
Q = np.diag([100.0, 50.0, 10000.0, 200.0, 0.0, 0.0]) 
R = np.diag([50.0, 50.0]) 
nx = 6
nu = 2

# --- 4. LQR 模型矩陣函式 (嚴格依照 251128.py 邏輯) ---
def get_model_matrix(M, m, r, I, l, Jp, D, Jd, g, delta_t):
    Term_wheel = 2 * m + 2 * I / (r**2)
    Qeq = Jp * M + (Jp + M * (l**2)) * Term_wheel 
    
    A23 = - (M**2) * l * g / Qeq
    A43 = M * g * (M + Term_wheel) / Qeq

    B21 = B22 = (Jp + M * (l**2) + M * l * r) / (r * Qeq) 
    B41_term = M * l * r + M + 2 * m + 2 * I / (r**2)
    B41 = B42 = - B41_term / (r * Qeq) 

    # --- 轉向 (Yaw) 動力學 ---
    B61_deno = r * (m + I / (r**2) + 2 * Jd / (D**2))
    
    # [依指示維持 251128.py 原本的邏輯]
    B61 = 1.0 / B61_deno   # 原始設定: 左輪輸入對 Yaw 產生正影響
    B62 = -B61             # 原始設定: 右輪輸入對 Yaw 產生負影響

    Ac = np.zeros((nx, nx))
    Ac[0, 1] = 1.0  
    Ac[1, 2] = A23  
    Ac[2, 3] = 1.0  
    Ac[3, 2] = A43  
    Ac[4, 5] = 1.0  

    Bc = np.zeros((nx, 2))
    Bc[1, 0] = B21; Bc[1, 1] = B22 
    Bc[3, 0] = B41; Bc[3, 1] = B42 
    Bc[5, 0] = B61; Bc[5, 1] = B62 
    
    A = np.eye(nx) + delta_t * Ac
    B = delta_t * Bc
    return A, B

def solve_DARE(A, B, Q, R, maxiter=150, eps=0.01):
    P = Q
    for i in range(maxiter):
        Pn = A.T @ P @ A - A.T @ P @ B @ inv(R + B.T @ P @ B) @ B.T @ P @ A + Q
        if (abs(Pn - P)).max() < eps: break
        P = Pn
    return Pn

def dlqr(A, B, Q, R):
    P = solve_DARE(A, B, Q, R)
    K = inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    return K

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

# --- Main Simulation ---
NUM_MOTOR = 6
# 請確認 xml 路徑是否正確
model = mujoco.MjModel.from_xml_path('/home/gino4/mujoco/mujoco_course/crazydog_urdf/urdf/scene.xml')
data = mujoco.MjData(model)

# 腿部目標角度 [L_hip, L_knee, L_wheel, R_hip, R_knee, R_wheel]
# 根據 test.xml 的 joint range 設定一個微蹲姿勢
target_dof_pos = np.array([1.27, -2.127, 0, 1.27, -2.127, 0])
target_dof_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# 設定模擬步長
simulation_dt = 0.001
model.opt.timestep = simulation_dt

# --- [新增] 模擬時間同步開關（False: 盡可能快；True: 近似即時）---
REALTIME = False

# --- [新增] 繪圖更新頻率/歷史長度（加速模擬用）---
PLOT_EVERY_N_STEPS = 50     # 想更快可調大（例如 100/200）
HISTORY_SECONDS = 300.0     # 想保留更久可調大
PLOT_DT = PLOT_EVERY_N_STEPS * simulation_dt
MAX_HISTORY = max(500, int(HISTORY_SECONDS / max(PLOT_DT, 1e-9)))

# PD Gain (腿部剛性設定)
# [L_hip, L_knee, L_wheel, R_hip, R_knee, R_wheel]
# 輪子 (idx 2, 5) 的 kp 設為 0，完全交給 LQR
kps = np.array([100.0, 100.0, 0.0, 100.0, 100.0, 0.0])
kds = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])  # [調整] 增加阻尼，抑制小震動

# 初始化 LQR
A_mat, B_mat = get_model_matrix(M, m, r, I, l, Jp, D, Jd, g, simulation_dt)
K_lqr = dlqr(A_mat, B_mat, Q, R)
print("LQR Gain K:\n", K_lqr)

# --- [新增] 輪扭矩一階低通（抑制高頻抖動；不限制輪速）---
torque_alpha = 0.3  # 0~1；越小越平滑但反應越慢
prev_tau_l = 0.0
prev_tau_r = 0.0

robot_x = 0.0 

x_ref_inited = False
x_ref = 0.0

# --- 站立定位外迴路：用「位置/速度」產生小角度的 pitch 參考 ---
# 設計重點：
# 1) 不改動原本可平衡的 LQR（Q/R 保持原設定），只把「目標 pitch」從 0 改成 theta_ref。
# 2) theta_ref 必須很小，否則容易飽和或翻倒。
Kp_x = 0.0
Kd_x = 0.0
theta_ref_max = np.deg2rad(0.0)

# --- [新增] Matplotlib 繪圖初始化 ---
plt.ion()  # 開啟互動模式

# --- [新增] 圖表顯示範圍設定 ---
time_window = 6.0          # x 軸顯示最近幾秒（rolling window）
pitch_ylim = (-0.25, 0.25)  # Pitch 顯示範圍（rad）
speed_ylim = (-5.0, 5.0)    # v_robot 顯示範圍（m/s）

pitchdot_ylim = (-8.0, 8.0)  # Pitch 角速度顯示範圍（rad/s）
torque_ylim = (-13.2, 13.2)  # 扭矩顯示範圍（Nm），對應 clip(-12,12) 略放寬


# 四張圖：pitch、pitch rate、速度（v_body）、扭矩（左右輪）
fig, (ax_pitch, ax_pitchdot, ax_v, ax_tau) = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
plt.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.08, hspace=0.25)

# Pitch 圖
line_pitch, = ax_pitch.plot([], [], 'r-', label='Pitch (rad)')
line_ref,   = ax_pitch.plot([], [], 'g--', label='Pitch Ref (rad)')
ax_pitch.set_ylim(*pitch_ylim)
ax_pitch.set_ylabel('Pitch (rad)')
ax_pitch.grid(True)
ax_pitch.legend()

# Pitch 角速度圖（theta_dot）
line_pitchdot, = ax_pitchdot.plot([], [], 'y-', label='Pitch rate (rad/s)')
ax_pitchdot.set_ylim(*pitchdot_ylim)
ax_pitchdot.set_ylabel('Pitch rate (rad/s)')
ax_pitchdot.grid(True)
ax_pitchdot.legend()

# 速度圖（v_body）
line_v,     = ax_v.plot([], [], 'b-', label='v_body (m/s)')
line_v_ref, = ax_v.plot([], [], 'k--', label='Zero Ref')
ax_v.set_ylim(*speed_ylim)
ax_v.set_ylabel('Speed (m/s)')
ax_v.grid(True)
ax_v.legend()

# 扭矩圖（左右輪）
line_tau_l, = ax_tau.plot([], [], 'm-', label='tau_L (Nm)')
line_tau_r, = ax_tau.plot([], [], 'c-', label='tau_R (Nm)')
ax_tau.set_ylim(*torque_ylim)
ax_tau.set_xlabel('Time (s)')
ax_tau.set_ylabel('Torque (Nm)')
ax_tau.grid(True)
ax_tau.legend()

# --- 初始 x 軸顯示範圍（避免一開始只顯示 0~0.04s）---
ax_pitch.set_xlim(0.0, time_window)
ax_pitchdot.set_xlim(0.0, time_window)
ax_v.set_xlim(0.0, time_window)
ax_tau.set_xlim(0.0, time_window)

# 數據儲存容器
history_time = []
history_pitch = []
history_pitch_dot = []
history_v = []
history_tau_l = []
history_tau_r = []
loop_count = 0

# --- [新增] Phase portrait：theta vs theta_dot（散點 + 時間漸變色）---
fig_phase, ax_phase = plt.subplots(1, 1, figsize=(6, 5))
phase_scatter = ax_phase.scatter([], [], c=[], s=8, cmap='viridis')
cbar_phase = fig_phase.colorbar(phase_scatter, ax=ax_phase)
cbar_phase.set_label('Time (s)')
ax_phase.set_xlabel('theta (rad)')
ax_phase.set_ylabel('theta_dot (rad/s)')
ax_phase.grid(True)
ax_phase.set_xlim(*pitch_ylim)
ax_phase.set_ylim(*pitchdot_ylim)
# -----------------------------------
try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
    
        while viewer.is_running():
            
            step_start = time.time()
            
            # --- 讀取感測器 ---
            # 根據 test.xml 順序: [Hip, Knee, Wheel]
            q_meas  = data.sensordata[0:6]   
            dq_meas = data.sensordata[6:12]  
            
            imu_quat = data.sensordata[18:22]    
            imu_gyro = data.sensordata[22:25]    

            w, x, y, z = imu_quat
            
            # 在讀完 imu_quat / imu_gyro 後加上
            frame_pos     = data.sensordata[28:31]   # [x,y,z] in world
            frame_lin_vel = data.sensordata[31:34]   # [vx,vy,vz] in world
            frame_ang_vel = data.sensordata[34:37]   # [wx,wy,wz] in world

            # --- 與 final_test.py 一致的座標系定義 ---
            # x：以「世界座標的 x 軸」作為位移參考；但位移增量採用輪速積分（驅動端一致性較高）。
            # 注意：這裡只在初始化時把 robot_x 錨定到 frame_pos[0]，後續以輪速積分更新。
            if not x_ref_inited:
                robot_x = frame_pos[0]
                x_ref = robot_x
                x_ref_inited = True
            
            # Roll, Pitch, Yaw 計算
            sinr_cosp = 2.0 * (w * x + y * z)
            cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)

            sinp = 2.0 * (w * y - z * x)
            if np.abs(sinp) >= 1.0: pitch = np.sign(sinp) * (np.pi / 2.0)
            else: pitch = np.arcsin(sinp)

            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            v_body = frame_lin_vel[0] * np.cos(yaw) + frame_lin_vel[1] * np.sin(yaw)

            # --- 速度（用輪速，比 frame_lin_vel 更貼近驅動端；也與原本可平衡版本一致） ---
            wheel_vel_l = dq_meas[2]
            wheel_vel_r = dq_meas[5]
            v_robot = (wheel_vel_l + wheel_vel_r) * 0.5 * r

            # 依照 final_test.py 的做法：用輪速積分更新 x（同一座標系的前進位移）
            robot_x += v_robot * simulation_dt

            # --- 站立定位外迴路：把「要回到原地」轉成一個小的 pitch 目標 ---
            # ex > 0 代表機器人在 x_ref 右邊（偏前），需要往回走，因此 theta_ref 取負（往後傾）。
            ex = x_ref - robot_x
            theta_ref = np.clip(Kp_x * ex + Kd_x * (0.0 - v_robot), -theta_ref_max, theta_ref_max)

            # --- LQR 狀態向量 ---
            theta_p = pitch
            theta_p_dot = imu_gyro[1]

            # Yaw 仍保留在狀態中（權重為 0，不會破壞原本可平衡設定）
            delta = yaw
            delta_dot = imu_gyro[2]

            x_state = np.array([
                robot_x,      # x
                v_robot,      # x_dot
                theta_p,      # theta
                theta_p_dot,  # theta_dot
                delta,        # delta (Yaw)
                delta_dot     # delta_dot (Yaw Rate)
            ])

            # --- 計算控制量 ---
            #u_lqr = -K_lqr @ x_state 
            # 只把 pitch 參考從 0 改成 theta_ref，其它參考維持原本設定
            x_state_ref = np.array([x_ref, 0.0, theta_ref, 0.0, 0.0, 0.0])
            u_lqr = -K_lqr @ (x_state - x_state_ref)
            
            # 根據 test.xml 馬達限制 (-12, 12) 進行裁切
            tau_l_wheel = np.clip(u_lqr[0], -12, 12)
            tau_r_wheel = np.clip(u_lqr[1], -12, 12)

            # --- [新增] 扭矩平滑（抑制高頻抖動；不限制輪速）---
            tau_l_wheel = torque_alpha * tau_l_wheel + (1.0 - torque_alpha) * prev_tau_l
            tau_r_wheel = torque_alpha * tau_r_wheel + (1.0 - torque_alpha) * prev_tau_r
            prev_tau_l, prev_tau_r = tau_l_wheel, tau_r_wheel

            # --- 混合控制 ---
            # 1. 腿部 PD 控制
            tau = pd_control(target_dof_pos, q_meas, kps, target_dof_vel, dq_meas, kds)
            
            # 2. 輪子 LQR 控制
            tau[2] = tau_l_wheel # Left Wheel
            tau[5] = tau_r_wheel # Right Wheel

            data.ctrl[:] = tau
            
            mujoco.mj_step(model, data)
            viewer.sync()

            # --- [新增] 繪圖更新邏輯 ---
            loop_count += 1
            # 每 PLOT_EVERY_N_STEPS 步更新一次圖表（加速模擬）
            if loop_count % PLOT_EVERY_N_STEPS == 0:
                history_time.append(data.time)
                history_pitch.append(pitch)
                history_pitch_dot.append(theta_p_dot)
                history_v.append(v_body)
                history_tau_l.append(tau_l_wheel)
                history_tau_r.append(tau_r_wheel)

                # 為了效能，只保留最近 MAX_HISTORY 點數據
                if len(history_time) > MAX_HISTORY:
                    history_time.pop(0)
                    history_pitch.pop(0)
                    history_pitch_dot.pop(0)
                    history_v.pop(0)
                    history_tau_l.pop(0)
                    history_tau_r.pop(0)

                # 更新線條數據
                line_pitch.set_data(history_time, history_pitch)
                line_ref.set_data(history_time, [theta_ref] * len(history_time))
                line_pitchdot.set_data(history_time, history_pitch_dot)
                line_v.set_data(history_time, history_v)
                line_v_ref.set_data(history_time, [0.0] * len(history_time))
                line_tau_l.set_data(history_time, history_tau_l)
                line_tau_r.set_data(history_time, history_tau_r)

                # x 軸：從 0 開始，右界隨時間延伸（保留前面資料）
                t_now = history_time[-1]
                x_right = max(time_window, t_now)
                ax_pitch.set_xlim(0.0, x_right)
                ax_pitchdot.set_xlim(0.0, x_right)
                ax_v.set_xlim(0.0, x_right)
                ax_tau.set_xlim(0.0, x_right)

                # 主圖更新（用 draw_idle 降低重繪成本）
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

                # Phase portrait（散點 + 時間漸變色）
                phase_scatter.set_offsets(np.column_stack([history_pitch, history_pitch_dot]))
                phase_scatter.set_array(np.array(history_time))
                if len(history_time) >= 2:
                    phase_scatter.set_clim(history_time[0], history_time[-1])
                fig_phase.canvas.draw_idle()
                fig_phase.canvas.flush_events()
            # -------------------------

            # 時間同步
            if REALTIME:
                time_until_next_step = simulation_dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
except KeyboardInterrupt:
    out_img = f"mujoco_plot_{time.strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(out_img, dpi=300)
    out_img_phase = out_img.replace('.png', '_phase.png')
    fig_phase.savefig(out_img_phase, dpi=300)
    print(f"\n[INFO] KeyboardInterrupt: saved plot to {out_img}")
    print(f"[INFO] KeyboardInterrupt: saved phase portrait to {out_img_phase}")
# 結束後關閉繪圖模式
plt.ioff()
plt.show()
