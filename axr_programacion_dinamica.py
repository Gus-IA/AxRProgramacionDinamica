import numpy as np
import matplotlib.pyplot as plt


def plot_env(v, a=None, title="$v_k$"):
    v = v.reshape(4, 4)
    if a is None:
        R, T = np.ones((4, 4)), np.ones((4, 4))
        L, B = -1.0 * R, -1.0 * T
    else:
        T = a[:, 0].reshape((4, 4))
        B = -1 * a[:, 1].reshape((4, 4))
        R = a[:, 2].reshape((4, 4))
        L = -1 * a[:, 3].reshape((4, 4))
    R[0, 0], R[3, 3] = 0, 0
    T[0, 0], T[3, 3] = 0, 0
    B[0, 0], B[3, 3] = 0, 0
    L[0, 0], L[3, 3] = 0, 0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    zeros = np.zeros((4, 4))
    ax1.matshow(v, cmap=plt.cm.tab20c)
    ax2.matshow(v, cmap=plt.cm.tab20c)
    for i in range(4):
        for j in range(4):
            c = v[j, i]
            if (i == 0 and j == 0) or (i == 3 and j == 3):
                ax1.text(
                    i, j, "X", va="center", ha="center", fontsize=20, color="black"
                )
            else:
                ax1.text(
                    i,
                    j,
                    f"{round(c,1)}",
                    va="center",
                    ha="center",
                    fontsize=20,
                    color="black",
                )
    ax2.quiver(L, zeros, scale=10)
    ax2.quiver(R, zeros, scale=10)
    ax2.quiver(zeros, T, scale=10)
    ax2.quiver(zeros, B, scale=10)
    ax1.axis(False)
    ax1.set_title(title, fontsize=20)
    ax2.axis(False)
    ax2.set_title("$\pi$", fontsize=20)
    plt.show()


estados = np.arange(16)
plot_env(estados, title="Estados")


def init_env():
    probas_transiciones = np.zeros((16, 4, 16))
    recompensas = -1 * np.ones((16, 4, 16))  # todas a -1

    # acciones: 0 (arriba), 1 (abajo), 2 (derecha), 3(izquierda)
    for s in range(16):
        # arriba
        if s < 4:
            probas_transiciones[s, 0, s] = 1
        else:
            probas_transiciones[s, 0, s - 4] = 1
        # abajo
        if s < 12:
            probas_transiciones[s, 1, s + 4] = 1
        else:
            probas_transiciones[s, 1, s] = 1
        # derecha
        if (s + 1) % 4 == 0:
            probas_transiciones[s, 2, s] = 1
        else:
            probas_transiciones[s, 2, s + 1] = 1
        # izquierda
        if s % 4 == 0:
            probas_transiciones[s, 3, s] = 1
        else:
            probas_transiciones[s, 3, s - 1] = 1

    # estados terminales
    probas_transiciones[0, :, :] = 0
    probas_transiciones[15, :, :] = 0
    recompensas[0, :, :] = 0
    recompensas[15, :, :] = 0

    return probas_transiciones, recompensas


probas_transiciones, recompensas = init_env()


pi = 0.25 * np.ones((16, 4))


def eval_pol(pi, gamma=1, its=100, v=None):
    v = np.zeros(16) if v is None else v
    for it in range(its):
        v_prev = v.copy()
        for s in range(16):
            v[s] = np.sum(
                [
                    pi[s, a]
                    * np.sum(
                        [
                            probas_transiciones[s][a][sp]
                            * (recompensas[s][a][sp] + gamma * v_prev[sp])
                            for sp in range(16)
                        ]
                    )
                    for a in range(4)
                ]
            )
    return v


v = eval_pol(pi, its=0)
plot_env(v, title="$v_0$")

v = eval_pol(pi, its=1)
plot_env(v, title="$v_1$")

v = eval_pol(pi, its=2)
plot_env(v, title="$v_2$")

v = eval_pol(pi, its=10)
plot_env(v, title="$v_{10}$")

v = eval_pol(pi, its=1000)
plot_env(v, title="$v_\pi$")


def eval_q(v, gamma=1):
    q = np.zeros((16, 4))
    for s in range(16):
        for a in range(4):
            q[s, a] = np.sum(
                [
                    probas_transiciones[s][a][sp]
                    * (recompensas[s][a][sp] + gamma * v[sp])
                    for sp in range(16)
                ]
            )
    return q


def greedy_pol(q):
    p = np.zeros((16, 4))
    p[range(16), q.argmax(axis=1)] = 1
    return p


q = eval_q(v)
pi = greedy_pol(q)

v = eval_pol(pi, its=1000)
plot_env(v, pi, title="$v_\pi$")

v = eval_pol(pi, its=1)  # value iteration !!!
q = eval_q(v)
pi = greedy_pol(q)

plot_env(v, pi, title="$v_1$")

v = eval_pol(pi, its=1, v=v)  # value iteration !!!
q = eval_q(v)
pi = greedy_pol(q)

plot_env(v, pi, title="$v_2$")

v = eval_pol(pi, its=1, v=v)  # value iteration !!!
q = eval_q(v)
pi = greedy_pol(q)

plot_env(v, pi, title="$v_3$")
