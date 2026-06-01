"""
Общее расчётное ядро для многослойных диэлектрических структур.
Здесь задаются матрица слоя, матрица всей стопки и коэффициенты
отражения/пропускания.
"""

import numpy as np


# Среды слева и справа от структуры. В работе это воздух.
N_IN = 1.0
N_OUT = 1.0

# Расчетная длина волны. Все толщины в UI задаются в долях lambda_0.
LAMBDA_0 = 1.0

# Значения по умолчанию для слоев:
# H — слой с большим показателем, L — слой с меньшим,
# D — дефектный слой в центре структуры.
DEFAULT_N_H = 2.0
DEFAULT_N_L = 1.25
DEFAULT_N_D = 2.0


def parse_number(text: str) -> float:
    """Позволяет вводить числа как 0.125, 0,125 или дроби вида 1/8."""
    text = text.strip().replace(",", ".")

    if "/" in text:
        # Дробная запись удобна для толщин вроде lambda_0/8.
        numerator, denominator = text.split("/", maxsplit=1)
        return float(numerator) / float(denominator)

    return float(text)


def optical_thickness(n: float, h: float, lambda_0: float = LAMBDA_0) -> float:
    """Оптическая толщина слоя: n*h/lambda_0."""
    return n * h / lambda_0


def is_quarter_wave(value: float) -> bool:
    """Проверка условия четвертьволнового слоя: n*h/lambda_0 = 1/4."""
    return abs(value - 0.25) < 1e-3


def is_half_wave(value: float) -> bool:
    """Проверка условия полуволнового слоя: n*h/lambda_0 = 1/2."""
    return abs(value - 0.5) < 1e-3


def layer_matrix(n: float, h: float, wavelength: float) -> np.ndarray:
    """Характеристическая матрица одного слоя."""
    # Фазовый набег слоя: delta = 2*pi*n*h/wavelength.
    delta = 2 * np.pi * n * h / wavelength

    # Матрица слоя связывает поля на левой и правой границах слоя.
    M_layer = np.array(
        [
            [np.cos(delta), 1j * np.sin(delta) / n],
            [1j * n * np.sin(delta), np.cos(delta)],
        ],
        dtype=complex,
    )

    return M_layer


def stack_matrix(layers: list[tuple[float, float]], wavelength: float) -> np.ndarray:
    """Полная матрица структуры: перемножение матриц слоев."""
    # Каждый слой хранится как пара (n, h):
    # n — показатель преломления, h — геометрическая толщина.
    M_total = np.eye(2, dtype=complex)

    for n, h in layers:
        # Порядок умножения соответствует порядку прохождения слоев.
        M_total = M_total @ layer_matrix(n, h, wavelength)

    return M_total


def reflection_transmission(
    layers: list[tuple[float, float]],
    wavelength: float,
    n_in: float = N_IN,
    n_out: float = N_OUT,
) -> tuple[float, float]:
    """Амплитуды r, t и интенсивности R, T для всей структуры."""
    M_total = stack_matrix(layers, wavelength)

    # Элементы полной характеристической матрицы:
    # M_total = [[a, b], [c, d]].
    a, b = M_total[0, 0], M_total[0, 1]
    c, d = M_total[1, 0], M_total[1, 1]

    # Из полной матрицы получаем амплитудные коэффициенты r и t.
    # Эти формулы учитывают согласование со средами n_in и n_out.
    denominator = n_in * a + n_in * n_out * b + c + n_out * d
    r = (n_in * a + n_in * n_out * b - c - n_out * d) / denominator
    t = 2 * n_in / denominator

    # Интенсивности: R = |r|^2, T = (n_out/n_in)*|t|^2.
    R = abs(r) ** 2
    T = (n_out / n_in) * abs(t) ** 2

    return float(R), float(T)


def make_period_layers(
    n_H: float,
    n_L: float,
    h_H: float,
    h_L: float,
    first_layer: str = "H",
) -> list[tuple[float, float]]:
    """Один период: HL или LH."""
    # H обычно означает слой с большим n, L — с меньшим n.
    if first_layer == "L":
        return [(n_L, h_L), (n_H, h_H)]

    return [(n_H, h_H), (n_L, h_L)]


def bragg_stack(
    periods: int,
    n_H: float,
    n_L: float,
    h_H: float,
    h_L: float,
    first_layer: str = "H",
) -> list[tuple[float, float]]:
    """Брэгговская стопка: (HL)^N или (LH)^N."""
    # Сначала собираем один период, затем повторяем его periods раз.
    period_layers = make_period_layers(n_H, n_L, h_H, h_L, first_layer)
    layers = []

    for _ in range(periods):
        layers.extend(period_layers)

    return layers


def defect_stack(
    periods: int,
    n_H: float,
    n_L: float,
    n_D: float,
    h_H: float,
    h_L: float,
    h_D: float,
) -> list[tuple[float, float]]:
    """Дефектная структура: (HL)^N D (LH)^N."""
    # Два брэгговских зеркала стоят по бокам от дефектного слоя D.
    left = bragg_stack(periods, n_H, n_L, h_H, h_L, first_layer="H")
    defect = [(n_D, h_D)]
    right = bragg_stack(periods, n_H, n_L, h_H, h_L, first_layer="L")

    return left + defect + right
