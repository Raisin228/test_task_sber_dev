"""Модель MLP. Атрошенко Б. С."""

import torch
import torch.nn as nn


class Network(nn.Module):
    """
    Нейросеть для ф-ии Q(s, a).

    Вход: состояние агента.
    Выход:
    """

    def __init__(self, state_size=4, hidden_size=64, action_size=4):
        """
        Инициализатор.

        :param state_size: кол-во входных параметров.
        :param hidden_size: кол-во нейронов в скрытом слое.
        :param action_size: кол-во выходных параметров.
        """

        super(Network, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state: torch.Tensor):
        """
        Прямой проход: принимает состояние, возвращает Q-значения.

        :param state: тензор с координатами агента.
        """

        return self.network(state)
