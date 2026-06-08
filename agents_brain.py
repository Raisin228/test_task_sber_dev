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

    def push(self, state, action, reward, next_state, done) -> None:
        """Сохраняю один переход в буфер."""

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size) -> tuple:
        """Достаю случайный батч переходов."""

        batch = random.sample(self.buffer, batch_size)

        # batch — список кортежей прошлых ходов
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Сам агент."""

    def __init__(
            self,
            state_size=2,
            action_size=4,
            lr=1e-3,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            batch_size=32,
            target_update_freq=50,
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done = 0  # счётчик шагов для target update

        self.main_network = Network(state_size, action_size)
        self.target_network = Network(state_size, action_size)

        self.target_network.load_state_dict(self.main_network.state_dict())

        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()  # ф-ия потерь
        self.buffer = ReplayBuffer()

    def act(self, state):
        """
        Что сделать агенту прямо сейчас?

        Вероятностью epsilon — случайное действие, иначе — действие с максимальным Q-значением.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        # Прогон через сеть и выбор макс. вероятного.
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.main_network(state_tensor)

        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """Сохранение перехода."""

        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        """Обновляю веса main_network на одном батче из буфера."""

        # Копим достаточное кол-во данных в буфере
        if len(self.buffer) < self.batch_size:
            return

        # достал батчи и получил предсказание для каждого хода
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # сеть предсказывает Quality для всех
        q_values = self.main_network(states)
        q_values = q_values.gather(
            dim=1,
            index=actions.unsqueeze(1)
        ).squeeze(1)

        # формула Беллмана
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1).values

            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # вычисление ф-ии потерь
        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        # считаю градиенты назад и обновляем веса
        loss.backward()
        self.optimizer.step()

        # обновляем target
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())

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
