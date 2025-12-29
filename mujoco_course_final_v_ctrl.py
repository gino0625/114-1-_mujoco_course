import mujoco
import mujoco.viewer
import time
import numpy as np
import matplotlib.pyplot as plt # [新增] 引入繪圖庫
from matplotlib.widgets import Slider, Button  # [新增] 速度控制 UI
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
# 狀態向量定義： [x, x_dot, theta(pitch), theta_dot, delta(yaw), delta_dot]
#
# 站立/定點：保留原本可穩定站立的權重。
Q_hold = np.diag([100.0, 20.0, 10000.0, 100.0, 0.0, 0.0])
R_hold = np.diag([5.0, 5.0])
#
# 速度控制：提高 x_dot 權重、適度放寬 theta 權重，讓系統可在不翻倒的前提下維持速度。
#（如果你想更「追速度」，可再提高 Q_speed[1,1]；如果覺得太容易前傾，可提高 Q_speed[2,2] 或 R_speed。）
Q_speed = np.diag([0.0, 250.0, 4000.0, 200.0, 0.0, 0.0])
R_speed = np.diag([9.0, 9.0])

# --- 3a. 速度控制參數 ---
V_CMD_MAX = 2.0            # 速度指令上限 (m/s)
V_CMD_SLEW = 2.0           # 速度指令斜率限制 (m/s^2)，避免一步到位造成翻倒
V_STOP_EPS = 0.01          # 視為「停止」的速度門檻 (m/s)
# 你目前的需求是「不需要對 yaw 角做任何限制」：
# - False：不強制左右輪扭矩相等（允許自然 yaw 漂移/轉向；仍然只有前後速度指令）
# - True ：強制左右輪扭矩相等（等同禁止差速轉向）
FORCE_EQUAL_WHEEL_TORQUE = True

# --- 3b. 輪速->線速度濾波/符號自動判定 ---
V_LPF_ALPHA = 0.25         # 0~1，越小越平滑
AUTO_WHEEL_SIGN = True     # True: 用 base 的前向速度估計輪速正負號
WHEEL_SIGN_MIN_SPD = 0.05  # 估計輪速符號所需的最小速度 (m/s)
WHEEL_SIGN_CONFIRM = 30    # 連續判定次數門檻
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

# --- XML 載入（給多個候選路徑，方便在不同環境直接跑） ---
xml_candidates = [
    '/home/gino4/mujoco/mujoco_course/crazydog_urdf/urdf/scene.xml',
    './scene.xml',
    '/mnt/data/scene.xml',
]
last_err = None
model = None
for p in xml_candidates:
    try:
        model = mujoco.MjModel.from_xml_path(p)
        print(f"[INFO] Loaded XML: {p}")
        break
    except Exception as e:
        last_err = e
if model is None:
    raise RuntimeError(f"Failed to load scene.xml from candidates: {xml_candidates}\nLast error: {last_err}")

data = mujoco.MjData(model)

# 腿部目標角度 [L_hip, L_knee, L_wheel, R_hip, R_knee, R_wheel]
# 根據 test.xml 的 joint range 設定一個微蹲姿勢
target_dof_pos = np.array([1.27, -2.127, 0, 1.27, -2.127, 0])
target_dof_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# 設定模擬步長
simulation_dt = 0.001
model.opt.timestep = simulation_dt

# PD Gain (腿部剛性設定)
# [L_hip, L_knee, L_wheel, R_hip, R_knee, R_wheel]
# 輪子 (idx 2, 5) 的 kp 設為 0，完全交給 LQR
kps = np.array([100.0, 100.0, 0.0, 100.0, 100.0, 0.0])
kds = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])  # [調整] 增加阻尼，抑制小震動

# 初始化 LQR（站立/定點 與 速度控制 各一組）
A_mat, B_mat = get_model_matrix(M, m, r, I, l, Jp, D, Jd, g, simulation_dt)
K_hold = dlqr(A_mat, B_mat, Q_hold, R_hold)
K_speed = dlqr(A_mat, B_mat, Q_speed, R_speed)
print("LQR Gain K_hold:\n", K_hold)
print("LQR Gain K_speed:\n", K_speed)

# 取得輪子馬達扭矩極限（以 actuator ctrlrange 為準）
WHEEL_ACT_L = 2
WHEEL_ACT_R = 5
wheel_tau_limit = float(min(model.actuator_ctrlrange[WHEEL_ACT_L, 1], model.actuator_ctrlrange[WHEEL_ACT_R, 1]))

# --- [新增] 輪扭矩一階低通（抑制高頻抖動；不限制輪速）---
torque_alpha = 0.3  # 0~1；越小越平滑但反應越慢
prev_tau_l = 0.0
prev_tau_r = 0.0

robot_x = 0.0

# --- 位置參考：
#   - 停止（Hold）模式：x_ref 固定，用於「停下來後站穩且不漂移」。
#   - 移動（Speed）模式：x_ref 以 v_ref 積分更新，避免 x 誤差累積與速度追蹤互相打架。
x_ref_inited = False
x_ref = 0.0

# --- 速度指令（由 UI 改寫 v_cmd_target；實際 v_cmd 會做斜率限制） ---
v_cmd_target = 0.0
v_cmd = 0.0
move_mode = False
stop_latch = True
brake_active = False   # [新增] Zero 後進入煞車流程

# --- 輪速->線速度低通與正負號估計 ---
wheel_sign = 1.0
wheel_sign_votes = 0
v_robot_f = 0.0
v_body_f = 0.0  # [新增] 機體前向速度低通（僅用於繪圖顯示）

# --- [新增] Matplotlib 繪圖初始化 ---
plt.ion()  # 開啟互動模式

# --- [新增] 圖表顯示範圍設定 ---
time_window = 20.0           # x 軸顯示最近幾秒（rolling window）
pitch_ylim = (-0.25, 0.25)  # Pitch 顯示範圍（rad）
pitchdot_ylim = (-5.0, 5.0)  # Pitch 角速度顯示範圍（rad/s）
_vylim = max(1.2, V_CMD_MAX * 1.3)
speed_ylim = (-_vylim, _vylim)  # v_robot / v_ref 顯示範圍（m/s）

tau_ylim = (-wheel_tau_limit * 1.1, wheel_tau_limit * 1.1)  # wheel torque 顯示範圍（Nm）


# 兩張圖：上 = pitch，下 = 前進速度 v_robot
fig, (ax_pitch, ax_pitchdot, ax_v, ax_tau) = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.22, hspace=0.25)

# Pitch 圖
line_pitch, = ax_pitch.plot([], [], 'r-', label='Pitch (rad)')
line_ref,   = ax_pitch.plot([], [], 'g--', label='Pitch Ref (rad)')
ax_pitch.set_ylim(*pitch_ylim)
ax_pitch.set_ylabel('Pitch (rad)')
ax_pitch.grid(True)
ax_pitch.legend()

# Pitch 角速度圖 (theta_dot)
line_pitchdot, = ax_pitchdot.plot([], [], 'y-', label='Pitch rate (rad/s)')
ax_pitchdot.set_ylim(*pitchdot_ylim)
ax_pitchdot.set_ylabel('Pitch rate (rad/s)')
ax_pitchdot.grid(True)
ax_pitchdot.legend()

# 速度圖（v_body）
line_v,     = ax_v.plot([], [], 'b-', label='v_body (m/s)')
line_v_ref, = ax_v.plot([], [], 'k--', label='v_ref (m/s)')
ax_v.set_ylim(*speed_ylim)
ax_v.set_xlabel('Time (s)')
ax_v.set_ylabel('Speed (m/s)')
ax_v.grid(True)
ax_v.legend()

# 扭矩圖（左右輪）
line_tau_l, = ax_tau.plot([], [], 'm-', label='tau_L (Nm)')
line_tau_r, = ax_tau.plot([], [], 'c-', label='tau_R (Nm)')
ax_tau.set_ylim(*tau_ylim)
ax_tau.set_xlabel('Time (s)')
ax_tau.set_ylabel('Torque (Nm)')
ax_tau.grid(True)
ax_tau.legend()

# --- [修正] 初始 x 軸顯示範圍（避免一開始只顯示 0~0.04s）---
ax_pitch.set_xlim(0.0, time_window)
ax_pitchdot.set_xlim(0.0, time_window)
ax_v.set_xlim(0.0, time_window)
ax_tau.set_xlim(0.0, time_window)

# 數據儲存容器
history_time = []
history_pitch = []
history_pitch_dot = []
history_v = []
history_v_ref = []
history_tau_l = []
history_tau_r = []
loop_count = 0

# --- [新增] 繪圖效能/歷史資料設定（僅影響繪圖，不影響控制） ---
PLOT_EVERY_N_STEPS = 50       # 每 N 步更新一次圖表（加速模擬）
HISTORY_SECONDS = 300.0       # 最多保留多少秒的繪圖資料（避免資料太久導致效能下降）
MAX_HISTORY = int(HISTORY_SECONDS / (PLOT_EVERY_N_STEPS * simulation_dt))

# --- 速度控制 UI（Slider + Buttons） ---
ui = {
    'stop_requested': False,
}

ax_slider = plt.axes([0.12, 0.12, 0.68, 0.03])
speed_slider = Slider(ax_slider, 'v_ref (m/s)', -V_CMD_MAX, V_CMD_MAX, valinit=0.0)

ax_btn_zero = plt.axes([0.82, 0.115, 0.14, 0.04])
btn_zero = Button(ax_btn_zero, 'Zero (Stop)')

ax_btn_p = plt.axes([0.12, 0.05, 0.12, 0.045])
btn_plus = Button(ax_btn_p, '+0.1')
ax_btn_m = plt.axes([0.25, 0.05, 0.12, 0.045])
btn_minus = Button(ax_btn_m, '-0.1')

status_text = ax_pitch.text(0.02, 0.92, '', transform=ax_pitch.transAxes)


# --- [新增] Phase Portrait：theta vs theta_dot（僅可視化，不影響控制） ---
# 以散點圖 + 漸變色（類似 heat map）表示時間前後順序
fig_phase, ax_phase = plt.subplots(1, 1, figsize=(5.5, 5.5))
# 以 NaN 當作不可見的初始化點，避免空資料時 colorbar/mappable 異常
phase_scatter = ax_phase.scatter([np.nan], [np.nan], c=[0.0], cmap='viridis', s=8, marker='o')
cbar_phase = fig_phase.colorbar(phase_scatter, ax=ax_phase)
cbar_phase.set_label('Time (s)')
ax_phase.set_xlim(*pitch_ylim)
ax_phase.set_ylim(*pitchdot_ylim)
ax_phase.set_xlabel('theta (rad)')
ax_phase.set_ylabel('theta_dot (rad/s)')
ax_phase.grid(True)


def _set_v_cmd_target(val: float):
    global v_cmd_target
    v_cmd_target = float(np.clip(val, -V_CMD_MAX, V_CMD_MAX))

def _on_slider_change(val):
    _set_v_cmd_target(val)

def _on_zero(_event):
    ui['stop_requested'] = True
    speed_slider.set_val(0.0)

def _on_plus(_event):
    _set_v_cmd_target(v_cmd_target + 0.1)
    speed_slider.set_val(v_cmd_target)

def _on_minus(_event):
    _set_v_cmd_target(v_cmd_target - 0.1)
    speed_slider.set_val(v_cmd_target)

speed_slider.on_changed(_on_slider_change)
btn_zero.on_clicked(_on_zero)
btn_plus.on_clicked(_on_plus)
btn_minus.on_clicked(_on_minus)
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

            # --- 速度（用輪速為主；必要時自動推定輪速正負號，避免「前後方向相反」） ---
            wheel_vel_l = dq_meas[2]
            wheel_vel_r = dq_meas[5]
            v_wheel_raw = (wheel_vel_l + wheel_vel_r) * 0.5 * r

            # --- [修正] 機體前向速度（用 frame_lin_vel 投影到機體前向；僅用於繪圖顯示與輪速符號判定） ---
            v_body_raw = frame_lin_vel[0] * np.cos(yaw) + frame_lin_vel[1] * np.sin(yaw)
            v_body_f = V_LPF_ALPHA * v_body_raw + (1.0 - V_LPF_ALPHA) * v_body_f
            v_body = v_body_f

            # 以「基地（IMU）」的世界線速度投影到機體前向（由 yaw 決定）做符號判定（僅用來判斷 wheel_sign）
            if AUTO_WHEEL_SIGN and (abs(wheel_sign_votes) < WHEEL_SIGN_CONFIRM):
                if (abs(v_wheel_raw) > WHEEL_SIGN_MIN_SPD) and (abs(v_body_raw) > WHEEL_SIGN_MIN_SPD):
                    if np.sign(v_wheel_raw) == np.sign(v_body_raw):
                        wheel_sign_votes += 1
                    else:
                        wheel_sign_votes -= 1
                    if abs(wheel_sign_votes) >= WHEEL_SIGN_CONFIRM:
                        wheel_sign = 1.0 if wheel_sign_votes > 0 else -1.0
                        print(f"[INFO] wheel_sign determined: {wheel_sign:+.0f} (votes={wheel_sign_votes})")

            v_robot_raw = wheel_sign * v_wheel_raw
            v_robot_f = V_LPF_ALPHA * v_robot_raw + (1.0 - V_LPF_ALPHA) * v_robot_f
            v_robot = v_robot_f

            # 用輪速積分更新 x（同一座標系的前進位移；與你的可平衡版本一致）
            robot_x += v_robot * simulation_dt

            # --- 速度指令處理（Stop / Speed 兩種模式） ---
            if ui['stop_requested']:
                v_cmd_target = 0.0
                v_cmd = 0.0               # [修正] Zero 時立即清除殘留速度命令（避免因斜率限制導致停不下來）
                ui['stop_requested'] = False
                stop_latch = False        # [修正] 讓煞車結束後可以重新鎖定 x_ref
                brake_active = True       # [新增] Zero 後先用 SPEED 把速度真的煞到接近 0
                x_ref = robot_x           # [修正] Zero 當下鎖住參考位置（煞車期間不積分）

            # 斜率限制：避免速度指令一步到位導致翻倒
            dv = np.clip(v_cmd_target - v_cmd, -V_CMD_SLEW * simulation_dt, V_CMD_SLEW * simulation_dt)
            v_cmd += dv

            # --- [修正] Zero 後煞車：在速度真正接近 0 前，一律用 SPEED（v_ref=0）煞車，不提早切回 HOLD ---
            if brake_active:
                move_mode = True
                v_ref = 0.0
                # x_ref 在煞車期間維持 Zero 當下位置（不積分）
                if abs(v_body) <= V_STOP_EPS:
                    brake_active = False
                    x_ref = robot_x
                    stop_latch = True
                    move_mode = False
                    v_ref = 0.0
            else:
                if abs(v_cmd) <= V_STOP_EPS:
                    # 停止模式：把 x_ref 鎖在當下位置，並用 K_hold 抑制漂移
                    if not stop_latch:
                        x_ref = robot_x
                        stop_latch = True
                    move_mode = False
                    v_ref = 0.0
                else:
                    # 速度模式：x_ref 以 v_ref 積分更新，讓系統以速度追蹤為主
                    if stop_latch:
                        x_ref = robot_x
                        stop_latch = False
                    move_mode = True
                    v_ref = float(np.clip(v_cmd, -V_CMD_MAX, V_CMD_MAX))
                    x_ref += v_ref * simulation_dt

            theta_ref = 0.0  # 速度模式下由 K_speed 的 Q/R 自行允許小幅前傾/後傾來維持速度

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
            if brake_active:
                x_state[1] = -v_body  # 煞車期間用 v_body 作為速度回授

            # --- 計算控制量 ---
            K_use = K_speed if move_mode else K_hold
            x_state_ref = np.array([x_ref, v_ref, theta_ref, 0.0, 0.0, 0.0])
            u_lqr = -K_use @ (x_state - x_state_ref)

            if FORCE_EQUAL_WHEEL_TORQUE:
                u_mean = 0.5 * (u_lqr[0] + u_lqr[1])
                u_lqr[0] = u_mean
                u_lqr[1] = u_mean

            tau_l_wheel = np.clip(u_lqr[0], -wheel_tau_limit, wheel_tau_limit)
            tau_r_wheel = np.clip(u_lqr[1], -wheel_tau_limit, wheel_tau_limit)

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
            # 每 PLOT_EVERY_N_STEPS 步更新一次圖表（僅影響繪圖更新頻率）
            if loop_count % PLOT_EVERY_N_STEPS == 0:
                history_time.append(data.time)
                history_pitch.append(pitch)
                history_pitch_dot.append(theta_p_dot)
                history_v.append(-v_body)  # [修正] 繪圖速度乘上負號
                history_v_ref.append(v_ref)
                history_tau_l.append(tau_l_wheel)
                history_tau_r.append(tau_r_wheel)

                # 為了效能，只保留最近 MAX_HISTORY 點數據
                if len(history_time) > MAX_HISTORY:
                    history_time.pop(0)
                    history_pitch.pop(0)
                    history_pitch_dot.pop(0)
                    history_v.pop(0)
                    history_v_ref.pop(0)
                    history_tau_l.pop(0)
                    history_tau_r.pop(0)

                # 更新線條數據
                line_pitch.set_data(history_time, history_pitch)
                line_pitchdot.set_data(history_time, history_pitch_dot)
                line_ref.set_data(history_time, [theta_ref] * len(history_time))
                line_v.set_data(history_time, history_v)
                line_v_ref.set_data(history_time, history_v_ref)
                line_tau_l.set_data(history_time, history_tau_l)
                line_tau_r.set_data(history_time, history_tau_r)


                # --- [新增] Phase Portrait 更新：theta vs theta_dot（散點 + 時間漸變色）---
                if len(history_pitch) > 0:
                    phase_offsets = np.column_stack((history_pitch, history_pitch_dot))
                    phase_scatter.set_offsets(phase_offsets)
                    # 用時間作為顏色（越晚越亮/深），形成時間順序漸變
                    t_arr = np.asarray(history_time, dtype=float)
                    phase_scatter.set_array(t_arr)
                    if len(t_arr) >= 2:
                        phase_scatter.set_clim(t_arr[0], t_arr[-1])
                fig_phase.canvas.draw_idle()
                fig_phase.canvas.flush_events()

                status_text.set_text(
                    f"mode={'SPEED' if move_mode else 'HOLD'} | "
                    f"v_cmd_target={v_cmd_target:+.2f} | v_ref={v_ref:+.2f} | v={v_body:+.2f} | "
                    f"wheel_sign={wheel_sign:+.0f}"
                )

                # --- x 軸顯示：從 0 開始，右界隨時間延伸（保留前面資料）---
                t_now = history_time[-1]
                ax_v.set_xlim(0.0, max(time_window, t_now))
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            # -------------------------

            # 時間同步
            time_until_next_step = simulation_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
except KeyboardInterrupt:
    out_img = f"mujoco_plot_{time.strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(out_img, dpi=300)
    print(f"\n[INFO] KeyboardInterrupt: saved plot to {out_img}")
# 結束後關閉繪圖模式
plt.ioff()
plt.show()
