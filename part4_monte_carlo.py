import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================
# Функція симуляції (з bandit2.py)
# ==========================================

def simulate_bandits(
    K, T, theta_true, sigma_sq=1.0,
    m0=0.0, tau_sq0=1.0,
    epsilon=0.1, seed=None
):
    theta_true = np.asarray(theta_true, dtype=float)
    sigma = np.sqrt(sigma_sq)
    algorithms = ['greedy', 'epsilon_greedy', 'thompson']

    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(len(algorithms))

    results = {}

    for alg, child_seed in zip(algorithms, child_seeds):
        rng = np.random.default_rng(child_seed)

        counts = np.zeros(K, dtype=int)
        empirical_means = np.zeros(K)

        m = np.full(K, m0, dtype=float)
        v = np.full(K, tau_sq0, dtype=float)

        chosen_arms = []

        for t in range(1, T + 1):
            if t <= K:
                chosen_arm = t - 1
            elif alg == 'greedy':
                chosen_arm = np.argmax(empirical_means)
            elif alg == 'epsilon_greedy':
                if rng.random() < epsilon:
                    chosen_arm = rng.integers(0, K)
                else:
                    chosen_arm = np.argmax(empirical_means)
            elif alg == 'thompson':
                sampled_theta = rng.normal(m, np.sqrt(v))
                chosen_arm = np.argmax(sampled_theta)

            reward = rng.normal(theta_true[chosen_arm], sigma)
            chosen_arms.append(chosen_arm)

            counts[chosen_arm] += 1
            empirical_means[chosen_arm] += (
                reward - empirical_means[chosen_arm]
            ) / counts[chosen_arm]

            if alg == 'thompson':
                precision_old = 1.0 / v[chosen_arm]
                precision_new = precision_old + 1.0 / sigma_sq
                v[chosen_arm] = 1.0 / precision_new
                m[chosen_arm] = v[chosen_arm] * (
                    precision_old * m[chosen_arm] + reward / sigma_sq
                )

        results[alg] = np.array(chosen_arms)

    return results


# ==========================================
# Частина 4.3 — Монте-Карло (M=1000 запусків)
# ==========================================

K = 2
T = 1000
M = 1000
theta = np.array([0.0, 0.3])
theta_star = np.max(theta)

algorithms = ['greedy', 'epsilon_greedy', 'thompson']

# Масив для збереження накопичених втрат: (M, T) для кожного алгоритму
regrets = {alg: np.zeros((M, T)) for alg in algorithms}

print(f"Запуск {M} симуляцій...")

for j in range(M):
    if j % 100 == 0:
        print(f"  Запуск {j}/{M}...")

    res = simulate_bandits(K=K, T=T, theta_true=theta, seed=j)

    for alg in algorithms:
        chosen_arms = res[alg]
        # R(t) = t*theta* - sum_{s=1}^{t} theta_{A_s}
        step_regrets = theta_star - theta[chosen_arms]
        regrets[alg][j] = np.cumsum(step_regrets)

print("Симуляції завершено!")

# ==========================================
# Усереднені накопичені втрати R_M(t)
# ==========================================

mean_regrets = {alg: regrets[alg].mean(axis=0) for alg in algorithms}

# Стандартне відхилення для довірчих смуг
std_regrets = {alg: regrets[alg].std(axis=0) for alg in algorithms}

# ==========================================
# Побудова графіка
# ==========================================

rounds = np.arange(1, T + 1)

colors = {
    'greedy': 'red',
    'epsilon_greedy': 'blue',
    'thompson': 'green'
}

labels = {
    'greedy': 'Pure Greedy',
    'epsilon_greedy': r'Epsilon-Greedy ($\varepsilon=0.1$)',
    'thompson': 'Thompson Sampling'
}

fig, ax = plt.subplots(figsize=(11, 6))

for alg in algorithms:
    mean = mean_regrets[alg]
    std = std_regrets[alg]
    color = colors[alg]

    ax.plot(rounds, mean, label=labels[alg], color=color, linewidth=2)

    # Довірча смуга ±1 std
    ax.fill_between(
        rounds,
        mean - std,
        mean + std,
        color=color,
        alpha=0.15
    )

ax.set_xlabel('Раунд $t$', fontsize=13)
ax.set_ylabel(r'Середні накопичені втрати $\bar{R}_M(t)$', fontsize=13)
ax.set_title(
    f'Частина 4.3. Монте-Карло порівняння алгоритмів\n'
    f'$K=2$, $\\theta=(0,\\,0.3)$, $T={T}$, $M={M}$ запусків',
    fontsize=13
)
ax.legend(fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig('part4_monte_carlo.png', dpi=200)
print("Збережено: part4_monte_carlo.png")

# ==========================================
# Фінальні значення R_M(T)
# ==========================================

print("\n--- Середні накопичені втрати R_M(T) ---")
for alg in algorithms:
    print(f"  {labels[alg]}: {mean_regrets[alg][-1]:.2f}")
