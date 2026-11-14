import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

class MultiArmedBandit:

    def __init__(self, true_means):
        self.true_means = np.array(true_means, dtype=float)
        self.n_arms = len(true_means)

        self.best_arm = np.argmax(true_means)
        self.optimal_mean = true_means[self.best_arm]

    def pull(self, arm):
        # Нормальний розподіл з дисперсією 1
        return np.random.normal(self.true_means[arm], 1.0)


def simulate_strategy(bandit, strategy_name, steps=1000, **kwargs):

    n_arms = bandit.n_arms

    counts = np.zeros(n_arms)
    values = np.zeros(n_arms)
    reward_sums = np.zeros(n_arms)

    cumulative_losses = np.zeros(steps)
    total_loss = 0.0

    # Апріорні параметри Thompson Sampling
    m0 = 0.0
    tau0_sq = 1.0
    sigma_sq = 1.0

    for t in range(steps):

        # Початкове дослідження: кожен важіль обирається один раз
        if t < n_arms:
            arm = t

        elif strategy_name == 'greedy':
            arm = np.argmax(values)

        elif strategy_name == 'epsilon_greedy':

            eps = kwargs.get('eps', 0.1)

            if np.random.rand() < eps:
                arm = np.random.choice(n_arms)
            else:
                arm = np.argmax(values)

        elif strategy_name == 'thompson':

            samples = np.zeros(n_arms)

            for a in range(n_arms):

                # Апостеріорна дисперсія
                tau_n_sq = 1 / (
                    1 / tau0_sq + counts[a] / sigma_sq
                )

                # Апостеріорне середнє
                m_n = tau_n_sq * (
                    m0 / tau0_sq +
                    reward_sums[a] / sigma_sq
                )

                # Випадкова вибірка
                samples[a] = np.random.normal(
                    m_n, np.sqrt(tau_n_sq)
                )

            arm = np.argmax(samples)

        else:
            raise ValueError("Невідома стратегія")

        # Отримання винагороди
        reward = bandit.pull(arm)

        # Оновлення статистики
        counts[arm] += 1
        reward_sums[arm] += reward

        values[arm] += (
            reward - values[arm]
        ) / counts[arm]

        # Накопичені втрати (Regret)
        step_loss = (
            bandit.optimal_mean -
            bandit.true_means[arm]
        )

        total_loss += step_loss
        cumulative_losses[t] = total_loss

    return cumulative_losses


# ==========================================
# ПАРАМЕТРИ МОДЕЛЮВАННЯ
# ==========================================

np.random.seed(42)

true_means = [0.0, 0.3]
steps = 1000

bandit = MultiArmedBandit(true_means)


# ==========================================
# ЗАПУСК ТРЬОХ АЛГОРИТМІВ
# ==========================================

loss_greedy = simulate_strategy(
    bandit, 'greedy', steps=steps
)

loss_eps = simulate_strategy(
    bandit, 'epsilon_greedy',
    steps=steps, eps=0.1
)

loss_thompson = simulate_strategy(
    bandit, 'thompson', steps=steps
)


# ==========================================
# ПОБУДОВА ГРАФІКА
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, steps + 1),
    loss_greedy,
    label='Pure Greedy',
    color='red'
)

plt.plot(
    range(1, steps + 1),
    loss_eps,
    label=r'Epsilon-Greedy ($\epsilon=0.1$)',
    color='blue'
)

plt.plot(
    range(1, steps + 1),
    loss_thompson,
    label='Thompson Sampling',
    color='green'
)

plt.xlabel('Номер раунду (t)')
plt.ylabel('Накопичені втрати R(t)')

plt.title(
    'Траєкторії накопичених втрат для одного запуску'
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("regret_plot.png", dpi=200, bbox_inches="tight")
print("График сохранён в regret_plot.png")
