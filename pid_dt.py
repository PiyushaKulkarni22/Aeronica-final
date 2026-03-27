# ================= PID REAL-TIME SIM (FIXED) =================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv, time
from collections import deque

def load_csv(file):
    data = []
    with open(file) as f:
        reader = csv.DictReader(f)
        for r in reader:
            data.append([float(r['roll']), float(r['pitch']), float(r['yaw'])])
    return np.array(data)

class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.int = 0
        self.prev_e = 0
        self.prev_d = 0
        self.alpha = 0.2
        self.i_lim = 200

    def compute(self, sp, meas, dt):
        if dt <= 0:
            return 0

        e = sp - meas

        p = self.kp * e

        self.int += e * dt
        self.int = np.clip(self.int, -self.i_lim, self.i_lim)
        i = self.ki * self.int

        d_raw = (e - self.prev_e) / dt
        d_f = self.alpha * d_raw + (1 - self.alpha) * self.prev_d
        d = self.kd * d_f

        self.prev_e, self.prev_d = e, d_f

        return p + i + d

class Mixer:
    def __init__(self):
        self.base = 1500

    def mix(self, u):
        r, p, y = u
        return np.clip([
            self.base - r + p - y,
            self.base + r + p + y,
            self.base - r - p + y,
            self.base + r - p - y
        ], 1000, 2000)

def run():
    data = load_csv("attitude_data_extreme.csv")

    mixer = Mixer()
    roll = PID(80, 10, 25)
    pitch = PID(80, 10, 25)
    yaw = PID(40, 5, 10)

    idx = 0
    prev = time.time()
    start_time = time.time()

    t_hist = deque(maxlen=200)
    pwm_hist = [deque(maxlen=200) for _ in range(4)]
    att_hist = [deque(maxlen=200) for _ in range(3)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    labels = ['M1 FL', 'M2 FR', 'M3 RL', 'M4 RR']
    colors = ['r', 'b', 'g', 'm']

    lines_pwm = [ax1.plot([], [], colors[i], label=labels[i])[0] for i in range(4)]
    lines_att = [
        ax2.plot([], [], 'r', label='Roll')[0],
        ax2.plot([], [], 'b', label='Pitch')[0],
        ax2.plot([], [], 'g', label='Yaw')[0]
    ]

    ax1.set_ylim(1000, 2000)
    ax1.set_title("PID PWM")
    ax1.legend()
    ax1.grid()

    ax2.set_ylim(-60, 60)
    ax2.set_title("Attitude")
    ax2.legend()
    ax2.grid()

    def update(frame):
        nonlocal idx, prev

        now = time.time()
        dt = now - prev
        prev = now

        if dt < 0.001:
            return lines_pwm + lines_att

        angles = data[idx]
        idx = (idx + 1) % len(data)

        u = np.array([
            roll.compute(0, angles[0], dt),
            pitch.compute(0, angles[1], dt),
            yaw.compute(0, angles[2], dt)
        ])

        pwm = mixer.mix(u)

        t_hist.append(now - start_time)

        for i in range(4):
            pwm_hist[i].append(pwm[i])

        for i in range(3):
            att_hist[i].append(np.degrees(angles[i]))

        t = list(t_hist)

        if len(t) > 2:
            for i in range(4):
                lines_pwm[i].set_data(t, list(pwm_hist[i]))

            for i in range(3):
                lines_att[i].set_data(t, list(att_hist[i]))

            ax1.set_xlim(t[0], t[-1])
            ax2.set_xlim(t[0], t[-1])

        return lines_pwm + lines_att

    ani = FuncAnimation(fig, update, interval=20, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()