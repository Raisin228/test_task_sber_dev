"""Обучение сетки. Атрошенко Б. С."""

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from world import GridWorld
from agents_brain import DQNAgent

WEIGHTS_FILE = "model_weights.pt"


def train(episodes=300):
    """Тренировка сетки."""

    env = GridWorld()
    agent = DQNAgent()

    if os.path.exists(WEIGHTS_FILE):
        saved_goal = agent.load(WEIGHTS_FILE)
        if saved_goal is None or tuple(saved_goal) == env.goal_pos:
            print(f"Найден файл весов '{WEIGHTS_FILE}' — обучение пропущено.")
            _demo(env, agent)
            return
        print(f"Цель изменилась {saved_goal} → {env.goal_pos} — обучаю заново.")
        os.remove(WEIGHTS_FILE)
        agent = DQNAgent()

    episode_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        total_reward = 0.0

        while not done:
            action = agent.act(state, valid_actions=env.valid_actions())
            next_state, reward, done = env.step(action)
            next_valid = env.valid_actions()  # валидные действия уже из нового положения
            agent.remember(state, action, reward, next_state, done, next_valid)
            agent.learn()
            state = next_state
            total_reward += reward
            steps += 1

            if steps >= 300:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        if (episode + 1) % 50 == 0:
            avg = np.mean(episode_rewards[-50:])
            avg_len = np.mean(episode_lengths[-50:])
            print(f"Эпизод {episode + 1}/{episodes} | "
                  f"Средняя награда (50 эпс.): {avg:.2f} | "
                  f"Средняя длина (50 эпс.): {avg_len:.1f} | "
                  f"ε = {agent.epsilon:.3f}")

    agent.save(WEIGHTS_FILE, goal_pos=env.goal_pos)
    print(f"Веса сохранены в '{WEIGHTS_FILE}'")
    _save_plot(episode_lengths)
    print("График сохранён в learned_policy.png")
    _demo(env, agent)


def _demo(env, agent):
    """Прогон обученного агента с отрисовкой каждого шага."""
    action_names = {0: "↑ вверх", 1: "↓ вниз", 2: "← лево", 3: "→ право"}

    print("\n=== Демонстрация обученного агента ===")

    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0  # чисто жадная политика

    state = env.reset()
    done = False
    step = 0
    total_reward = 0.0

    env.render()
    while not done and step < 20:
        action = agent.act(state, valid_actions=env.valid_actions())
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward
        step += 1
        print(f"Шаг {step}: {action_names[action]}, награда = {reward:+.1f}")
        env.render()
        time.sleep(0.3)

    if done:
        print(f"Цель достигнута за {step} шагов. Суммарная награда: {total_reward:.2f}")
    else:
        print(f"Цель не достигнута за {step} шагов. Суммарная награда: {total_reward:.2f}")

    agent.epsilon = saved_epsilon


def _save_plot(episode_lengths):
    window = 20
    smoothed = np.convolve(
        episode_lengths,
        np.ones(window) / window,
        mode="valid"
    )

    plt.figure(figsize=(10, 5))
    plt.plot(episode_lengths, color="lightgray", alpha=0.6, label="длина эпизода")
    plt.plot(
        range(window - 1, len(episode_lengths)),
        smoothed,
        color="steelblue",
        linewidth=2,
        label=f"среднее по {window} эпизодам",
    )
    plt.xlabel("Эпизод")
    plt.ylabel("Число шагов до цели")
    plt.title("Обучение DQN-агента — длина эпизода от итерации")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("learned_policy.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    train()
