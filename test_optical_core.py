"""
Тесты расчетного ядра optical_core.

Проверяются три вещи:
1) законы сохранения (R + T = 1 для прозрачной структуры),
2) совпадение с аналитическими формулами, известными из теории,
3) корректный разбор пользовательского ввода.

Запуск: pytest -q
"""

import numpy as np
import pytest

from optical_core import (
    bragg_stack,
    defect_stack,
    is_half_wave,
    is_quarter_wave,
    layer_matrix,
    make_period_layers,
    optical_thickness,
    parse_int,
    parse_number,
    reflection_transmission,
    stack_matrix,
)


# Параметры, на которых удобно сверяться с теорией.
N_H = 2.0
N_L = 1.25
H_H = 1.0 / (4 * N_H)  # четвертьволновой слой H
H_L = 1.0 / (4 * N_L)  # четвертьволновой слой L


# --------------------------------------------------------------------------
# Законы сохранения
# --------------------------------------------------------------------------


@pytest.mark.parametrize("periods", [1, 2, 5, 10])
@pytest.mark.parametrize("wavelength", [0.6, 0.85, 1.0, 1.3, 1.7])
def test_energy_is_conserved_in_bragg_stack(periods, wavelength):
    """Без поглощения вся энергия делится между отражением и пропусканием."""
    layers = bragg_stack(periods, N_H, N_L, H_H, H_L)
    R, T = reflection_transmission(layers, wavelength)

    assert R + T == pytest.approx(1.0, abs=1e-12)


def test_energy_is_conserved_in_defect_stack():
    layers = defect_stack(6, N_H, N_L, 2.0, H_H, H_L, 1.0 / (2 * 2.0))

    for wavelength in np.linspace(0.82, 1.18, 41):
        R, T = reflection_transmission(layers, wavelength)
        assert R + T == pytest.approx(1.0, abs=1e-12)


def test_energy_is_conserved_for_random_stacks():
    """Случайные (не четвертьволновые) структуры тоже должны быть прозрачны."""
    rng = np.random.default_rng(seed=20240101)

    for _ in range(50):
        count = int(rng.integers(1, 12))
        layers = [
            (float(rng.uniform(1.05, 3.5)), float(rng.uniform(0.02, 0.6)))
            for _ in range(count)
        ]
        wavelength = float(rng.uniform(0.5, 2.0))
        R, T = reflection_transmission(layers, wavelength)

        assert 0.0 <= R <= 1.0
        assert 0.0 <= T <= 1.0
        assert R + T == pytest.approx(1.0, abs=1e-10)


# --------------------------------------------------------------------------
# Сравнение с аналитическими результатами
# --------------------------------------------------------------------------


def test_empty_stack_is_fully_transparent():
    """Структуры нет — свет проходит целиком."""
    R, T = reflection_transmission([], 1.0)

    assert R == pytest.approx(0.0, abs=1e-14)
    assert T == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("n", [1.5, 2.0, 3.4])
def test_half_wave_layer_is_invisible(n):
    """Полуволновой слой (n*h = lambda/2) не отражает: 'absentee layer'."""
    R, T = reflection_transmission([(n, 0.5 / n)], 1.0)

    assert R == pytest.approx(0.0, abs=1e-12)
    assert T == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("n", [1.5, 2.0, 3.4])
def test_quarter_wave_layer_matches_single_film_formula(n):
    """Для четвертьволновой пленки в воздухе R = ((1 - n^2)/(1 + n^2))^2."""
    R, T = reflection_transmission([(n, 0.25 / n)], 1.0)
    expected = ((1 - n**2) / (1 + n**2)) ** 2

    assert R == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("periods", [1, 2, 4, 8])
def test_quarter_wave_mirror_matches_admittance_formula(periods):
    """Классическая формула для зеркала (HL)^N на lambda_0 в воздухе.

    Эффективный адмиттанс стопки Y = (n_H/n_L)^(2N), отсюда
    R = ((1 - Y) / (1 + Y))^2.
    """
    layers = bragg_stack(periods, N_H, N_L, H_H, H_L, first_layer="H")
    R, _ = reflection_transmission(layers, 1.0)

    Y = (N_H / N_L) ** (2 * periods)
    expected = ((1 - Y) / (1 + Y)) ** 2

    assert R == pytest.approx(expected, abs=1e-12)


def test_reflection_grows_with_number_of_periods():
    """Чем больше периодов, тем ближе R к единице на центральной длине волны."""
    values = [
        reflection_transmission(bragg_stack(n, N_H, N_L, H_H, H_L), 1.0)[0]
        for n in (1, 2, 4, 8, 12)
    ]

    assert values == sorted(values)
    assert values[-1] > 0.999


def test_defect_opens_transmission_peak_inside_stop_band():
    """Полуволновой дефект открывает узкий пик пропускания на lambda_0."""
    periods = 6
    h_D = 0.5 / 2.0

    with_defect = defect_stack(periods, N_H, N_L, 2.0, H_H, H_L, h_D)
    without_defect = bragg_stack(2 * periods, N_H, N_L, H_H, H_L)

    T_defect = reflection_transmission(with_defect, 1.0)[1]
    T_regular = reflection_transmission(without_defect, 1.0)[1]

    assert T_defect > 0.99
    assert T_regular < 1e-4


def test_defect_peak_is_narrow():
    """Пик дефектной моды должен быть узким: рядом с lambda_0 пропускание падает."""
    layers = defect_stack(6, N_H, N_L, 2.0, H_H, H_L, 0.5 / 2.0)

    assert reflection_transmission(layers, 1.0)[1] > 0.99
    assert reflection_transmission(layers, 1.05)[1] < 0.1


# --------------------------------------------------------------------------
# Свойства матриц
# --------------------------------------------------------------------------


def test_layer_matrix_determinant_is_one():
    """Матрица слоя унимодулярна: det M = 1."""
    for n, h, wavelength in [(1.5, 0.2, 1.0), (2.4, 0.07, 0.63), (1.0, 1.0, 1.31)]:
        assert np.linalg.det(layer_matrix(n, h, wavelength)) == pytest.approx(
            1.0, abs=1e-12
        )


def test_stack_matrix_determinant_is_one():
    layers = bragg_stack(5, N_H, N_L, H_H, H_L)
    assert np.linalg.det(stack_matrix(layers, 0.93)) == pytest.approx(1.0, abs=1e-10)


def test_stack_matrix_of_empty_stack_is_identity():
    assert np.allclose(stack_matrix([], 1.0), np.eye(2))


def test_zero_thickness_layer_is_identity():
    assert np.allclose(layer_matrix(2.0, 0.0, 1.0), np.eye(2))


# --------------------------------------------------------------------------
# Сборка структур
# --------------------------------------------------------------------------


def test_make_period_layers_respects_first_layer():
    assert make_period_layers(2.0, 1.25, 0.1, 0.2, "H") == [(2.0, 0.1), (1.25, 0.2)]
    assert make_period_layers(2.0, 1.25, 0.1, 0.2, "L") == [(1.25, 0.2), (2.0, 0.1)]


def test_bragg_stack_has_two_layers_per_period():
    assert len(bragg_stack(7, N_H, N_L, H_H, H_L)) == 14


def test_defect_stack_is_symmetric_around_defect():
    """(HL)^N D (LH)^N: структура должна быть зеркальной относительно дефекта."""
    periods = 3
    layers = defect_stack(periods, N_H, N_L, 1.7, H_H, H_L, 0.25)

    assert len(layers) == 4 * periods + 1

    middle = len(layers) // 2
    assert layers[middle] == (1.7, 0.25)
    assert layers[:middle] == layers[middle + 1 :][::-1]


# --------------------------------------------------------------------------
# Вспомогательные функции и разбор ввода
# --------------------------------------------------------------------------


def test_optical_thickness():
    assert optical_thickness(2.0, 0.125) == pytest.approx(0.25)
    assert optical_thickness(2.0, 0.25, lambda_0=2.0) == pytest.approx(0.25)


def test_quarter_and_half_wave_checks():
    assert is_quarter_wave(0.25)
    assert not is_quarter_wave(0.30)
    assert is_half_wave(0.5)
    assert not is_half_wave(0.25)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.125", 0.125),
        ("0,125", 0.125),
        ("  1/8  ", 0.125),
        ("1/8", 0.125),
        ("2", 2.0),
        ("-0.5", -0.5),
        ("1e-3", 0.001),
    ],
)
def test_parse_number_accepts_valid_input(text, expected):
    assert parse_number(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "   ", "abc", "1/0", "1/", "/8", "1//8", "nan"])
def test_parse_number_rejects_bad_input(text):
    """Любой мусор должен стать ValueError, а не упасть в UI трейсбеком."""
    with pytest.raises(ValueError):
        parse_number(text, name="толщина H")


def test_parse_number_error_message_names_the_field():
    with pytest.raises(ValueError, match="толщина H"):
        parse_number("abc", name="толщина H")


@pytest.mark.parametrize(("text", "expected"), [("6", 6), (" 12 ", 12), ("-3", -3)])
def test_parse_int_accepts_valid_input(text, expected):
    assert parse_int(text) == expected


@pytest.mark.parametrize("text", ["", "6.5", "abc", "1/2"])
def test_parse_int_rejects_bad_input(text):
    with pytest.raises(ValueError):
        parse_int(text, name="Число периодов N")
