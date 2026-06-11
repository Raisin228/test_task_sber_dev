"""Здесь хранится мозг агента, который принимает решение куда идти. Атрошенко Б. С."""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

from model import Network


class ReplayBuffer:
    """Буфер для хранения прошлых состояний агента."""

    def __init__(self, capacity=10_000):
        """Инициализатор."""

        # когда буфер заполнен — не нужно следить за размером. старые значения выбросятся сами.
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_valid) -> None:
        """Сохраняю один переход в буфер."""

        self.buffer.append((state, action, reward, next_state, done, next_valid))

    def sample(self, batch_size) -> tuple:
        """Достаю случайный батч переходов."""

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_valid_batch = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
            list(next_valid_batch),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Сам агент."""

    def __init__(
            self,
            state_size=4,
            action_size=4,
            lr=1e-3,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.985,
            batch_size=32,
            tau=0.005,
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau  # коэффициент мягкого обновления target-сети

        self.main_network = Network(state_size=state_size, action_size=action_size)
        self.target_network = Network(state_size=state_size, action_size=action_size)

        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss — устойчивее MSE при больших ошибках
        self.buffer = ReplayBuffer()

    def act(self, state, valid_actions=None):
        """
        Что сделать агенту прямо сейчас?

        Вероятностью epsilon — случайное валидное действие, иначе — действие с максимальным Q-значением
        среди валидных.
        """
        if random.random() < self.epsilon:
            pool = valid_actions if valid_actions is not None else list(range(self.action_size))
            return random.choice(pool)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.main_network(state_tensor).squeeze(0)

        if valid_actions is not None:
            mask = torch.full((self.action_size,), float('-inf'))
            for a in valid_actions:
                mask[a] = 0.0
            q_values = q_values + mask

        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done, next_valid=None) -> None:
        """Сохранение перехода вместе с валидными действиями из следующего состояния."""

        if next_valid is None:
            next_valid = list(range(self.action_size))
        self.buffer.push(state, action, reward, next_state, done, next_valid)

    def learn(self):
        """Обновляю веса main_network на одном батче из буфера."""

        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, next_valid_batch = self.buffer.sample(self.batch_size)

        # сеть предсказывает Quality для выбранных действий
        q_values = self.main_network(states)
        q_values = q_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

        # формула Беллмана с маскировкой недопустимых действий в следующем состоянии
        with torch.no_grad():
            next_q_all = self.target_network(next_states)  # (batch, action_size)
            for i, (done, valid) in enumerate(zip(dones.tolist(), next_valid_batch)):
                if done:
                    # терминальное состояние — значение следующего шага не нужно
                    next_q_all[i] = 0.0
                else:
                    mask = torch.full((self.action_size,), float('-inf'))
                    for a in valid:
                        mask[a] = 0.0
                    next_q_all[i] = next_q_all[i] + mask
            next_q_values = next_q_all.max(dim=1).values
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # мягкое обновление target-сети: θ_target ← τ·θ_main + (1-τ)·θ_target
        for main_p, target_p in zip(self.main_network.parameters(), self.target_network.parameters()):
            target_p.data.copy_(self.tau * main_p.data + (1 - self.tau) * target_p.data)

    def save(self, path: str, goal_pos=None) -> None:
        """Сохранить веса и конфигурацию цели."""
        torch.save(
            {"state_dict": self.main_network.state_dict(), "goal_pos": goal_pos},
            path,
        )

    def load(self, path: str):
        """
        Загрузить веса и синхронизировать target-сеть.

        :return: сохранённая позиция цели или None (для старого формата файла).
        """
        checkpoint = torch.load(path, weights_only=False)
        if isinstance(checkpoint, dict):
            self.main_network.load_state_dict(checkpoint["state_dict"])
            saved_goal = checkpoint.get("goal_pos")
        else:
            self.main_network.load_state_dict(checkpoint)
            saved_goal = None
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.epsilon = self.epsilon_min
        return saved_goal
