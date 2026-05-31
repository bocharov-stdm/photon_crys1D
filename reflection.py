"""
UI для спектра отражения многослойной брэгговской стопки.

Расчетное ядро находится в optical_core.py.
Структура: (HL)^N или (LH)^N.
"""

import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from optical_core import (
    DEFAULT_N_H,
    DEFAULT_N_L,
    LAMBDA_0,
    N_IN,
    N_OUT,
    bragg_stack,
    is_quarter_wave,
    optical_thickness,
    parse_number,
    reflection_transmission,
)


DEFAULT_POINTS = 800
DEFAULT_FIGURE_SIZE = (8.5, 5.2)
DEFAULT_H_H = LAMBDA_0 / (4 * DEFAULT_N_H)
DEFAULT_H_L = LAMBDA_0 / (4 * DEFAULT_N_L)


def optical_label(name: str, value: float) -> str:
    text = f"{name} = {value:.4f}"

    if is_quarter_wave(value):
        text += " — четвертьволновой слой"

    return text


class ReflectionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Отражение брэгговской стопки")

        self.entries: dict[str, tk.Entry] = {}
        self.first_layer = tk.StringVar(value="H")
        self.optical_H_text = tk.StringVar()
        self.optical_L_text = tk.StringVar()

        self.build_controls()
        self.build_plot()
        self.update_optical_labels()
        self.plot()

    def build_controls(self) -> None:
        panel = tk.Frame(self.root, padx=16, pady=16)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(
            panel,
            text="Отражение брэгговской стопки",
            font=("Arial", 14, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        tk.Label(
            panel,
            text=(
                "Структура строится из двух слоев H и L.\n"
                "Толщина задается как h/λ0: в долях\n"
                "расчетной длины волны λ0.\n"
                "N — число периодов. Один период содержит два слоя."
            ),
            justify=tk.LEFT,
            wraplength=300,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 14))

        fields = [
            ("h_H", "Геометрическая толщина слоя H, hH/λ0", f"{DEFAULT_H_H:g}"),
            ("h_L", "Геометрическая толщина слоя L, hL/λ0", f"{DEFAULT_H_L:g}"),
            ("n_H", "Показатель преломления nH", f"{DEFAULT_N_H:g}"),
            ("n_L", "Показатель преломления nL", f"{DEFAULT_N_L:g}"),
            ("periods", "Число периодов N", "2, 4, 8"),
            ("lambda_min", "Диапазон λ/λ0: от", "0.55"),
            ("lambda_max", "до", "1.55"),
            ("points", "Количество точек", str(DEFAULT_POINTS)),
        ]

        for row, (key, label, default) in enumerate(fields, start=2):
            tk.Label(panel, text=label).grid(row=row, column=0, sticky="w", pady=4)
            entry = tk.Entry(panel, width=16)
            entry.insert(0, default)
            entry.grid(row=row, column=1, sticky="ew", pady=4)
            entry.bind("<KeyRelease>", self.update_optical_labels)
            self.entries[key] = entry

        optical_row = len(fields) + 2
        tk.Label(panel, textvariable=self.optical_H_text, fg="#444444").grid(
            row=optical_row,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(8, 0),
        )
        tk.Label(panel, textvariable=self.optical_L_text, fg="#444444").grid(
            row=optical_row + 1,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(2, 10),
        )

        first_layer_frame = tk.LabelFrame(panel, text="Первый слой периода", padx=8, pady=6)
        first_layer_frame.grid(
            row=optical_row + 2,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(0, 8),
        )
        tk.Radiobutton(
            first_layer_frame,
            text="H: структура (HL)^N",
            variable=self.first_layer,
            value="H",
            command=self.plot,
        ).pack(anchor="w")
        tk.Radiobutton(
            first_layer_frame,
            text="L: структура (LH)^N",
            variable=self.first_layer,
            value="L",
            command=self.plot,
        ).pack(anchor="w")

        button_row = optical_row + 3
        tk.Button(panel, text="Построить график", command=self.plot).grid(
            row=button_row,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(12, 4),
        )
        tk.Button(panel, text="Сбросить параметры", command=self.reset).grid(
            row=button_row + 1,
            column=0,
            columnspan=2,
            sticky="ew",
        )

    def build_plot(self) -> None:
        plot_area = tk.Frame(self.root, padx=8, pady=8)
        plot_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Button(
            plot_area,
            text="Сохранить график",
            command=self.save_plot,
        ).pack(anchor="ne", pady=(0, 4))

        self.figure = Figure(figsize=DEFAULT_FIGURE_SIZE, dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_area)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def save_plot(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Сохранить график",
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("PDF document", "*.pdf"),
                ("SVG image", "*.svg"),
                ("All files", "*.*"),
            ],
        )

        if not path:
            return

        current_size = self.figure.get_size_inches()

        try:
            self.figure.set_size_inches(DEFAULT_FIGURE_SIZE, forward=False)
            self.figure.savefig(path, dpi=300, bbox_inches="tight")
        finally:
            self.figure.set_size_inches(current_size, forward=False)
            self.canvas.draw_idle()

    def float_value(self, key: str) -> float:
        return parse_number(self.entries[key].get())

    def int_value(self, key: str) -> int:
        return int(self.entries[key].get().strip())

    def periods(self) -> list[int]:
        raw = self.entries["periods"].get().replace(";", ",")
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]

        if not values:
            raise ValueError("Введите хотя бы одно значение N.")
        if any(value <= 0 for value in values):
            raise ValueError("Все значения N должны быть больше нуля.")

        return values

    def read_parameters(self) -> dict[str, float | int | list[int]]:
        values: dict[str, float | int | list[int]] = {
            "h_H": self.float_value("h_H"),
            "h_L": self.float_value("h_L"),
            "n_H": self.float_value("n_H"),
            "n_L": self.float_value("n_L"),
            "periods": self.periods(),
            "lambda_min": self.float_value("lambda_min"),
            "lambda_max": self.float_value("lambda_max"),
            "points": self.int_value("points"),
        }

        if values["h_H"] <= 0 or values["h_L"] <= 0:
            raise ValueError("Геометрические толщины слоев должны быть больше нуля.")
        if values["n_H"] <= 0 or values["n_L"] <= 0:
            raise ValueError("Показатели преломления должны быть больше нуля.")
        if values["points"] < 2:
            raise ValueError("Количество точек должно быть не меньше 2.")
        if values["lambda_min"] == values["lambda_max"]:
            raise ValueError("Начало и конец диапазона не должны совпадать.")

        if values["lambda_min"] > values["lambda_max"]:
            values["lambda_min"], values["lambda_max"] = (
                values["lambda_max"],
                values["lambda_min"],
            )

        return values

    def update_optical_labels(self, event: tk.Event | None = None) -> None:
        try:
            n_H = self.float_value("n_H")
            n_L = self.float_value("n_L")
            h_H = self.float_value("h_H")
            h_L = self.float_value("h_L")
        except ValueError:
            self.optical_H_text.set("nH*hH/λ0 = ...")
            self.optical_L_text.set("nL*hL/λ0 = ...")
            return

        self.optical_H_text.set(
            optical_label("nH*hH/λ0", optical_thickness(n_H, h_H, LAMBDA_0))
        )
        self.optical_L_text.set(
            optical_label("nL*hL/λ0", optical_thickness(n_L, h_L, LAMBDA_0))
        )

    def reset(self) -> None:
        defaults = {
            "h_H": f"{DEFAULT_H_H:g}",
            "h_L": f"{DEFAULT_H_L:g}",
            "n_H": f"{DEFAULT_N_H:g}",
            "n_L": f"{DEFAULT_N_L:g}",
            "periods": "2, 4, 8",
            "lambda_min": "0.55",
            "lambda_max": "1.55",
            "points": str(DEFAULT_POINTS),
        }

        for key, value in defaults.items():
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, value)

        self.first_layer.set("H")
        self.update_optical_labels()
        self.plot()

    def plot(self) -> None:
        try:
            values = self.read_parameters()
        except ValueError as error:
            messagebox.showerror("Ошибка ввода", str(error))
            return

        n_H = float(values["n_H"])
        n_L = float(values["n_L"])
        h_H = float(values["h_H"])
        h_L = float(values["h_L"])
        periods_list = values["periods"]
        lambda_min = float(values["lambda_min"])
        lambda_max = float(values["lambda_max"])
        points = int(values["points"])

        wavelengths = np.linspace(lambda_min * LAMBDA_0, lambda_max * LAMBDA_0, points)
        colors = ["#F28E2B", "#356AE6", "#D94E41", "#5AA469", "#8E5EA2"]

        self.ax.clear()
        self.update_optical_labels()

        for index, periods in enumerate(periods_list):
            layers = bragg_stack(
                periods=periods,
                n_H=n_H,
                n_L=n_L,
                h_H=h_H,
                h_L=h_L,
                first_layer=self.first_layer.get(),
            )
            R_values = [
                reflection_transmission(layers, wavelength, n_in=N_IN, n_out=N_OUT)[0]
                for wavelength in wavelengths
            ]

            self.ax.plot(
                wavelengths / LAMBDA_0,
                R_values,
                linewidth=1.05,
                color=colors[index % len(colors)],
                label=f"N = {periods}",
            )

        self.ax.set_title("Спектр отражения многослойной стопки")
        self.ax.set_xlabel(r"$\lambda/\lambda_0$")
        self.ax.set_ylabel("R — коэффициент отражения")
        self.ax.set_xlim(lambda_min, lambda_max)
        self.ax.set_ylim(0, 1.05)
        self.ax.grid(alpha=0.18)
        self.ax.legend(frameon=False, loc="lower right")

        self.figure.tight_layout()
        self.canvas.draw()


def main() -> None:
    root = tk.Tk()
    root.geometry("1120x640")
    ReflectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
