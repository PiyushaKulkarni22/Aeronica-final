# ================= FINAL LQR REAL-TIME SIM =================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv, time
from collections import deque

# ---------- LOAD CSV ----------
def load_csv(file):
    data = []
    with open(file) as f:
        reader = csv.DictReader(f)
        for r in reader:
            data.append([
                float(r['roll']),
                float(r['pitch']),
                float(r['yaw'])
            ])
    return np.array(data)


# ---------- LQR CONTROLLER ----------
class LQR:
    def __init__(self):
        self.K = np.array([
    [12, 0, 0, 2.0, 0, 0],
    [0, 12, 0, 0, 2.0, 0],
    [0, 0, 8, 0, 0, 1.5]
])
        self.prev = np.zeros(3)

    def compute(self, angles, dt):
        if dt <= 0:
            return np.zeros(3)

        rates = (angles - self.prev) / dt
        self.prev = angles.copy()

        x = np.concatenate([angles, rates])
        u = -self.K @ x

        return np.clip(u, -500, 500)


# ---------- MIXER ----------
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


# ---------- MAIN SIM ----------
def run():
    data = load_csv("attitude_data_extreme.csv")

    ctrl = LQR()
    mixer = Mixer()

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
    ax1.set_title("LQR PWM Output")
    ax1.legend()
    ax1.grid()

    ax2.set_ylim(-60, 60)
    ax2.set_title("Attitude (deg)")
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

        # LQR control
        u = ctrl.compute(angles, dt)

        u = u * 5
       

        pwm = mixer.mix(u)

        # normalized time
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


# ---------- RUN ----------
if __name__ == "__main__":
    run()