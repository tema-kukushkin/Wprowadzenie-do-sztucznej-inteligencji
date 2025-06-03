import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import asyncio
import nest_asyncio
import platform

# Zastosowanie nest_asyncio dla Jupyter
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Funkcja wybierająca akcję
def choose_action(state, q_table, epsilon, n_actions, strategy='epsilon_greedy', temperature=1.0):
    if strategy == 'epsilon_greedy':
        if np.random.random() < epsilon:
            return np.random.randint(n_actions)
        return np.argmax(q_table[state])
    elif strategy == 'softmax':
        q_values = q_table[state]
        exp_q = np.exp(q_values / temperature)
        probabilities = exp_q / np.sum(exp_q)
        return np.random.choice(n_actions, p=probabilities)

# Funkcja trenująca Q-learning
async def train_q_learning(env, episodes, alpha, gamma, epsilon_start, epsilon_end, epsilon_decay, strategy='epsilon_greedy', temperature=1.0):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))
    rewards = []
    epsilon = epsilon_start

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = choose_action(state, q_table, epsilon, n_actions, strategy, temperature)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Aktualizacja Q-table
            q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Kontrola pętli w Pyodide
        if platform.system() == "Emscripten":
            await asyncio.sleep(0.001)

    return q_table, rewards

# Funkcja testująca agenta
def test_agent(env, q_table, episodes=100):
    total_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards), np.std(total_rewards)

# Główna funkcja eksperymentu
async def run_experiments():
    env = gym.make("Taxi-v3")
    episodes = 1000
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995

    # Eksperyment 1: Wpływ alpha
    alphas = [0.1, 0.5, 0.9]
    results_alpha = {}
    for alpha in alphas:
        q_table, rewards = await train_q_learning(env, episodes, alpha, gamma, epsilon_start, epsilon_end, epsilon_decay)
        mean_reward, std_reward = test_agent(env, q_table)
        results_alpha[alpha] = {'rewards': rewards, 'mean_test_reward': mean_reward, 'std_test_reward': std_reward}

    # Eksperyment 2: Wpływ strategii eksploracji
    strategies = [('epsilon_greedy', None), ('softmax', 1.0), ('softmax', 0.1)]
    results_strategy = {}
    for strategy, temp in strategies:
        q_table, rewards = await train_q_learning(env, episodes, alpha=0.5, gamma=gamma, epsilon_start=epsilon_start,
                                                epsilon_end=epsilon_end, epsilon_decay=epsilon_decay,
                                                strategy=strategy, temperature=temp if temp else 1.0)
        mean_reward, std_reward = test_agent(env, q_table)
        results_strategy[f"{strategy}_temp{temp if temp else ''}"] = {'rewards': rewards, 'mean_test_reward': mean_reward, 'std_test_reward': std_reward}

    # Wykresy
    plt.figure(figsize=(12, 5))
    
    # Wykres 1: Wpływ alpha
    plt.subplot(1, 2, 1)
    for alpha in alphas:
        plt.plot(np.convolve(results_alpha[alpha]['rewards'], np.ones(50)/50, mode='valid'), label=f'alpha={alpha}')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.title('Wpływ współczynnika uczenia')
    plt.legend()

    # Wykres 2: Wpływ strategii
    plt.subplot(1, 2, 2)
    for key in results_strategy:
        plt.plot(np.convolve(results_strategy[key]['rewards'], np.ones(50)/50, mode='valid'), label=key)
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.title('Wpływ strategii eksploracji')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Analiza wyników
    print("Wpływ współczynnika uczenia (alpha):")
    for alpha in alphas:
        print(f"Alpha={alpha}: Średnia nagroda={results_alpha[alpha]['mean_test_reward']:.2f}, Odchylenie={results_alpha[alpha]['std_test_reward']:.2f}")

    print("\nWpływ strategii eksploracji:")
    for key in results_strategy:
        print(f"Strategia {key}: Średnia nagroda={results_strategy[key]['mean_test_reward']:.2f}, Odchylenie={results_strategy[key]['std_test_reward']:.2f}")

# Uruchomienie eksperymentów
async def main():
    try:
        await run_experiments()
    except Exception as e:
        print(f"Błąd podczas uruchamiania eksperymentów: {e}")

if platform.system() == "Emscripten":
    asyncio.ensure_future(run_experiments())
else:
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(run_experiments())
    except RuntimeError:
        asyncio.run(main())