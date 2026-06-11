"""Среда, действия агента. Атрошенко Б. С."""

import numpy as np
from numpy import ndarray


# Агент, Среда, Действие, Награда

class GridWorld:
    """Работа с игровым полем: инициализация, отрисовка, шаг."""

    def __init__(self):
        """Инициализатор."""

        self.grid_size = 5

        # S-образный лабиринт:
        #  A . . . .
        #  . . █ █ █
        #  . . . . .
        #  █ █ █ . .
        #  G . . . .
        self.start_pos = (0, 0)
        self.goal_pos = (4, 0)
        self.wall_pos = [(1, 2), (1, 3), (1, 4), (3, 0), (3, 1), (3, 2)]

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
                elif (row, col) in self.wall_pos:
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

    def valid_actions(self) -> list[int]:
        """Возвращает список действий, которые не ведут в стену или за границу."""
        result = []
        for action_id, (dr, dc) in self.actions.items():
            new_row = self.agent_pos[0] + dr
            new_col = self.agent_pos[1] + dc
            if (0 <= new_row < self.grid_size
                    and 0 <= new_col < self.grid_size
                    and (new_row, new_col) not in self.wall_pos):
                result.append(action_id)
        return result

    def step(self, action) -> tuple[ndarray, float, bool]:
        """
        Совершить некоторый шаг в среде.

        :return: положение агента, награду и индикатор - дошёл ли?
        """

        old_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])

        delta_row, delta_col = self.actions[action]
        new_row = self.agent_pos[0] + delta_row
        new_col = self.agent_pos[1] + delta_col

        # Вышли за границу поля
        if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
            return self._get_state(), -3.0, False

        # Врезался в стену
        if (new_row, new_col) in self.wall_pos:
            return self._get_state(), -3.0, False

        self.agent_pos = (new_row, new_col)

        # Дошёл до цели
        if self.agent_pos == self.goal_pos:
            return self._get_state(), +10.0, True

        # небольшой бонус за приближение к цели, штраф за удаление
        new_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        return self._get_state(), -0.1 + 0.1 * (old_dist - new_dist), False

    def _get_state(self) -> np.ndarray:
        """
        Возвращает состояние — вектор [agent_row, agent_col, goal_row, goal_col].
        Нормализуем на размер сетки.

        :return: положение агента и цели.
        """

        row, col = self.agent_pos
        goal_row, goal_col = self.goal_pos
        return np.array(
            [row / self.grid_size, col / self.grid_size,
             goal_row / self.grid_size, goal_col / self.grid_size],
            dtype=np.float32,
        )
