import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def simulate_bandits(
    K, T, theta_true, sigma_sq=1.0,
    m0=0.0, tau_sq0=1.0,
    epsilon=0.1, seed=None
):
    """
    Моделювання одного запуску трьох алгоритмів:
    Pure Greedy, Epsilon-Greedy, Thompson Sampling
    для нормального багаторукого бандита.
    """

    # Перевірка вхідних параметрів
    if not isinstance(K, (int, np.integer)) or isinstance(K, bool) or K <= 0:
        raise ValueError("K має бути додатним цілим числом.")

    if not isinstance(T, (int, np.integer)) or isinstance(T, bool) or T <= 0:
        raise ValueError("T має бути додатним цілим числом.")

    theta_true = np.asarray(theta_true, dtype=float)

    if theta_true.shape != (K,):
        raise ValueError("theta_true повинен містити рівно K значень.")

    if sigma_sq <= 0:
        raise ValueError("sigma_sq має бути додатним.")

    if tau_sq0 <= 0:
        raise ValueError("tau_sq0 має бути додатним.")

    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon має бути в межах від 0 до 1.")

    sigma = np.sqrt(sigma_sq)

    # Назви алгоритмів
    algorithms = ['greedy', 'epsilon_greedy', 'thompson']

    # Окремий генератор випадкових чисел для кожного алгоритму
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(len(algorithms))

    results = {}

    for alg, child_seed in zip(algorithms, child_seeds):

        rng = np.random.default_rng(child_seed)

        # Статистика для Greedy та Epsilon-Greedy
        counts = np.zeros(K, dtype=int)
        empirical_means = np.zeros(K)

        # Апостеріорні параметри Thompson Sampling
        m = np.full(K, m0, dtype=float)
        v = np.full(K, tau_sq0, dtype=float)

        # Історія
        chosen_arms = []
        rewards = []

        history_m = []
        history_v = []

        for t in range(1, T + 1):

            # Ініціалізація: перші K раундів обираємо кожну ручку
            if t <= K:
                chosen_arm = t - 1

            else:

                # Pure Greedy
                if alg == 'greedy':

                    chosen_arm = np.argmax(empirical_means)

                # Epsilon-Greedy
                elif alg == 'epsilon_greedy':

                    if rng.random() < epsilon:

                        # Дослідження: випадкова ручка
                        chosen_arm = rng.integers(0, K)

                    else:

                        # Експлуатація: найкраща ручка
                        chosen_arm = np.argmax(empirical_means)

                # Thompson Sampling
                elif alg == 'thompson':

                    # Вибірка з апостеріорних розподілів
                    sampled_theta = rng.normal(m, np.sqrt(v))

                    chosen_arm = np.argmax(sampled_theta)

            # Отримання винагороди
            reward = rng.normal(
                theta_true[chosen_arm],
                sigma
            )

            # Збереження історії
            chosen_arms.append(chosen_arm)
            rewards.append(reward)

            # Оновлення емпіричного середнього
            counts[chosen_arm] += 1

            empirical_means[chosen_arm] += (
                reward - empirical_means[chosen_arm]
            ) / counts[chosen_arm]

            # Байєсівське оновлення для Thompson Sampling
            if alg == 'thompson':

                precision_old = 1.0 / v[chosen_arm]

                precision_new = precision_old + 1.0 / sigma_sq

                v[chosen_arm] = 1.0 / precision_new

                m[chosen_arm] = v[chosen_arm] * (
                    precision_old * m[chosen_arm]
                    + reward / sigma_sq
                )

                # Збереження апостеріорних параметрів
                history_m.append(m.copy())
                history_v.append(v.copy())

        # Збереження результатів
        results[alg] = {
            'chosen_arms': np.array(chosen_arms),
            'rewards': np.array(rewards),
            'counts': counts,
            'history_m': np.array(history_m) if alg == 'thompson' else None,
            'history_v': np.array(history_v) if alg == 'thompson' else None
        }

    return results


# ==========================================
# 3.1. Перевірка для K = 3
# ==========================================

K_test = 3
T_test = 100
theta_test = np.array([-0.3, 0.2, 0.7])

res_test = simulate_bandits(
    K=K_test,
    T=T_test,
    theta_true=theta_test,
    seed=42
)

print("--- Результати перевірки (Частина 3.1) ---")

for alg_name, data in res_test.items():

    print("-" * 30)

    print(f"Алгоритм: {alg_name}")

    print(f"Кількість виборів кожної ручки: {data['counts']}")

    print(f"Загальна сума винагород: {np.sum(data['rewards']):.2f}")


# ==========================================
# 3.2. Побудова графіків для K = 2
# ==========================================

K_plot = 2
T_plot = 300

theta_plot = np.array([0.0, 0.3])

res_plot = simulate_bandits(
    K=K_plot,
    T=T_plot,
    theta_true=theta_plot,
    seed=123
)


# ==========================================
# (а) Обрані ручки залежно від раунду
# ==========================================

algorithms_display = [
    ('greedy', 'Pure Greedy'),
    ('epsilon_greedy', r'Epsilon-Greedy ($\varepsilon = 0.1$)'),
    ('thompson', 'Thompson Sampling')
]

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for ax, (alg_key, alg_title) in zip(axes, algorithms_display):

    ax.step(
        range(1, T_plot + 1),
        res_plot[alg_key]['chosen_arms'] + 1,
        where='mid',
        linewidth=1.5
    )

    ax.set_yticks([1, 2])

    ax.set_yticklabels([
        'Ручка 1 ($\\theta=0$)',
        'Ручка 2 ($\\theta=0.3$)'
    ])

    ax.set_ylabel(alg_title, fontsize=10)

    ax.set_ylim(0.5, 2.5)

    ax.grid(True, linestyle='--', alpha=0.6)

    ax.set_xlabel('Раунд ($t$)')

fig.suptitle(
    'Обрані ручки залежно від раунду для кожного алгоритму ($K=2, T=300$)',
    fontsize=14
)

plt.tight_layout()

plt.savefig('chosen_arms_plot.png', dpi=300)
print("Збережено: chosen_arms_plot.png")


# ==========================================
# (б) Апостеріорні середні Thompson Sampling
# ==========================================

thompson_m = res_plot['thompson']['history_m']

thompson_v = res_plot['thompson']['history_v']

rounds = np.arange(1, T_plot + 1)

plt.figure(figsize=(12, 6))

for arm in range(K_plot):

    m_vals = thompson_m[:, arm]

    v_vals = thompson_v[:, arm]

    std_vals = np.sqrt(v_vals)

    upper_bound = m_vals + 2 * std_vals

    lower_bound = m_vals - 2 * std_vals

    line, = plt.plot(
        rounds,
        m_vals,
        label=f'Апостеріорне середнє ручки {arm + 1}',
        linewidth=2
    )

    plt.fill_between(
        rounds,
        lower_bound,
        upper_bound,
        color=line.get_color(),
        alpha=0.2
    )

    # Істинне значення theta
    plt.axhline(
        y=theta_plot[arm],
        linestyle='--',
        color=line.get_color(),
        alpha=0.8,
        label=f'Істинне значення ручки {arm + 1}'
    )

plt.title('Апостеріорні середні та довірчі інтервали Thompson Sampling')

plt.xlabel('Раунд ($t$)')

plt.ylabel('Значення параметра')

plt.legend()

plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

plt.savefig('thompson_posterior_plot.png', dpi=300)
print("Збережено: thompson_posterior_plot.png")


# ==========================================
# Побудова графіків Частини 3.2
# ==========================================

fig = plt.figure(figsize=(16, 10))

gs = fig.add_gridspec(
    3, 2,
    width_ratios=[1.2, 1],
    hspace=0.5,
    wspace=0.25
)

# Ліворуч: три графіки вибору ручок
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])

# Праворуч: графік Thompson Sampling
ax4 = fig.add_subplot(gs[:, 1])

# ------------------------------------------
# (а) Обрані ручки залежно від раунду
# ------------------------------------------

algorithms = [
    ('greedy', 'Pure Greedy', ax1, 'blue'),
    ('epsilon_greedy', r'Epsilon-Greedy ($\epsilon=0.1$)', ax2, 'green'),
    ('thompson', 'Thompson Sampling', ax3, 'purple')
]

for alg, title, ax, color in algorithms:

    chosen = res_plot[alg]['chosen_arms'] + 1

    ax.step(
        np.arange(1, T_plot + 1),
        chosen,
        where='mid',
        color=color
    )

    ax.set_title(title)
    ax.set_ylabel('Обрана ручка')
    ax.set_yticks([1, 2])
    ax.set_ylim(0.5, 2.5)
    ax.grid(True, alpha=0.3)

    ax.set_xlabel('Раунд t')


# ------------------------------------------
# (б) Апостеріорні середні Thompson Sampling
# ------------------------------------------

m_history = res_plot['thompson']['history_m']
v_history = res_plot['thompson']['history_v']

rounds = np.arange(1, T_plot + 1)

for arm in range(K_plot):

    m_values = m_history[:, arm]
    v_values = v_history[:, arm]

    std = np.sqrt(v_values)

    color = ['blue', 'orange'][arm]

    # Апостеріорне середнє
    ax4.plot(
        rounds,
        m_values,
        color=color,
        label=f'Ручка {arm + 1}: $m_{{t}}$'
    )

    # Діапазон m ± 2√v
    ax4.fill_between(
        rounds,
        m_values - 2 * std,
        m_values + 2 * std,
        color=color,
        alpha=0.2
    )

    # Справжнє значення theta
    ax4.axhline(
        theta_plot[arm],
        color=color,
        linestyle='--',
        label=f'Справжнє $\\theta_{arm + 1}$'
    )

ax4.set_title('Thompson Sampling: апостеріорні середні')

ax4.set_xlabel('Раунд t')
ax4.set_ylabel('Значення параметра')

ax4.legend()
ax4.grid(True, alpha=0.3)


# ------------------------------------------
# Загальний заголовок
# ------------------------------------------

fig.suptitle(
    'Частина 3.2. Траєкторії навчання',
    fontsize=16
)

plt.tight_layout()

plt.savefig('part_3_2.png', dpi=300)
print("Збережено: part_3_2.png")
