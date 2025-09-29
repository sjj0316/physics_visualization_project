import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 삼중진자 파라미터
class TriplePendulumParams:
    def __init__(self, m1=1, m2=1, m3=1, l1=1, l2=1, l3=1, g=9.81):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.g = g

# 삼중진자 운동 방정식 (간단화, 실제론 수치적분 필요)
def derivs(state, params):
    # state: [theta1, omega1, theta2, omega2, theta3, omega3]
    # 실제론 매우 복잡함. 여기선 예시로 단순화
    theta1, omega1, theta2, omega2, theta3, omega3 = state
    m1, m2, m3 = params.m1, params.m2, params.m3
    l1, l2, l3 = params.l1, params.l2, params.l3
    g = params.g
    # 실제 삼중진자 운동 방정식은 sympy로 유도 필요
    domega1 = -g/l1 * np.sin(theta1)
    domega2 = -g/l2 * np.sin(theta2)
    domega3 = -g/l3 * np.sin(theta3)
    return np.array([omega1, domega1, omega2, domega2, omega3, domega3])

# 시뮬레이션
class TriplePendulumSim:
    def __init__(self, params, y0, dt=0.01):
        self.params = params
        self.y = np.array(y0)
        self.dt = dt
        self.history = [self.y.copy()]
    def step(self):
        k1 = derivs(self.y, self.params)
        k2 = derivs(self.y + 0.5*self.dt*k1, self.params)
        k3 = derivs(self.y + 0.5*self.dt*k2, self.params)
        k4 = derivs(self.y + self.dt*k3, self.params)
        self.y += (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        self.history.append(self.y.copy())
    def run(self, steps):
        for _ in range(steps):
            self.step()

# 2D 애니메이션

def animate_2d(sim: TriplePendulumSim, interval=20):
    history = np.array(sim.history)
    l1, l2, l3 = sim.params.l1, sim.params.l2, sim.params.l3
    x1 = l1 * np.sin(history[:,0])
    y1 = -l1 * np.cos(history[:,0])
    x2 = x1 + l2 * np.sin(history[:,2])
    y2 = y1 - l2 * np.cos(history[:,2])
    x3 = x2 + l3 * np.sin(history[:,4])
    y3 = y2 - l3 * np.cos(history[:,4])

    fig, ax = plt.subplots()
    ax.set_xlim(-l1-l2-l3-0.5, l1+l2+l3+0.5)
    ax.set_ylim(-l1-l2-l3-0.5, l1+l2+l3+0.5)
    ax.set_aspect('equal')
    line, = ax.plot([], [], 'o-', lw=2)

    def update(frame):
        thisx = [0, x1[frame], x2[frame], x3[frame]]
        thisy = [0, y1[frame], y2[frame], y3[frame]]
        line.set_data(thisx, thisy)
        return line,

    ani = FuncAnimation(fig, update, frames=len(history), interval=interval, blit=True)
    plt.show()

# 실행 예시
if __name__ == "__main__":
    params = TriplePendulumParams(m1=1, m2=1, m3=1, l1=1, l2=1, l3=1)
    y0 = [np.pi/2, 0, np.pi/2, 0, np.pi/2, 0]
    sim = TriplePendulumSim(params, y0, dt=0.02)
    sim.run(500)
    animate_2d(sim)
