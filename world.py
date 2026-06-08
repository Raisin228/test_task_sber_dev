"""Среда, действия агента. Атрошенко Б. С."""

import numpy as np
from numpy import ndarray


# Агент, Среда, Действие, Награда

class GridWorld:
    """Работа с игровым полем: инициализация, отрисовка, шаг."""

    def __init__(self):
        """Инициализатор."""

        self.grid_size = 5

        # Координаты агента, финиша и стены
        self.start_pos = (0, 4)
        self.goal_pos = (4, 0)
        self.wall_pos = (2, 2)

        self.agent_pos = self.start_pos

        self.actions = {
            0: (-1, 0),  # вверх
            1: (1, 0),  # вниз
            2: (0, -1),  # лево
            3: (0, 1)  # право
        }

    def render(self) -> None:
        """Вывести в консоль."""

        for row in range(self.grid_size):
            line = ""
            for col in range(self.grid_size):
                if (row, col) == self.agent_pos:
                    line += " A "
                elif (row, col) == self.goal_pos:
                    line += " G "
                elif (row, col) == self.wall_pos:
                    line += " █ "
                else:
                    line += " . "
            print(line)
        print()

    def reset(self) -> np.ndarray:
        """
        Возвращает агента в начальное состояние.

        :return положение агента в сетке.
        """

        self.agent_pos = self.start_pos
        return self._get_state()

    def step(self, action) -> tuple[ndarray, float, bool] | None:
        """
        Совершить некоторый шаг в среде.

        :return: положение агента, награду и индикатор - дошёл ли?
        """

        delta_row, delta_col = self.actions[action]
        new_row = self.agent_pos[0] + delta_row
        new_col = self.agent_pos[1] + delta_col

        # Вышли за границу поля
        if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
            return self._get_state(), -1.0, False

        # Врезался в стену
        if (new_row, new_col) == self.wall_pos:
            return self._get_state(), -1.0, False

        self.agent_pos = (new_row, new_col)

        # Дошёл до цели
        if self.agent_pos == self.goal_pos:
            return self._get_state(), +10.0, True

        # Не дошёл до цели - штрафной "пустой ход"
        return self._get_state(), -0.1, False

    def _get_state(self) -> np.ndarray:
        """
        Возвращает состояние - numpy-вектор [row, col].
        Нормализуем на размер сетки.

        :return: положение агента.
        """

        row, col = self.agent_pos
        return np.array([row / self.grid_size, col / self.grid_size], dtype=np.float32)
