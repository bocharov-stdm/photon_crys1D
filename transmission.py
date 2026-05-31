"""
UI для спектра пропускания структуры с дефектной пластинкой.

Расчетное ядро находится в optical_core.py.
Структура с дефектом: (HL)^N D (LH)^N.
"""

import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from optical_core import (
    DEFAULT_N_D,
    DEFAULT_N_H,
    DEFAULT_N_L,
    LAMBDA_0,
    bragg_stack,
    defect_stack,
    is_half_wave,
    is_quarter_wave,
    optical_thickness,
    parse_number,
    reflection_transmission,
)


DEFAULT_POINTS = 1200
DEFAULT_FIGURE_SIZE = (8.5, 5.2)
DEFAULT_H_H = LAMBDA_0 / (4 * DEFAULT_N_H)
DEFAULT_H_L = LAMBDA_0 / (4 * DEFAULT_N_L)
DEFAULT_H_D = LAMBDA_0 / (2 * DEFAULT_N_D)


def optical_label(name: str, value: float, layer_type: str = "layer") -> str:
    text = f"{name} = {value:.4f}"

    if is_quarter_wave(value):
        text += " — четвертьволновой слой"
    elif layer_type == "defect" and is_half_wave(value):
        text += " — полуволновой дефект"
    elif is_half_wave(value):
        text += " — полуволновой слой"

    return text


class TransmissionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Пропускание с дефектной пластинкой")

        self.entries: dict[str, tk.Entry] = {}
        self.show_regular = tk.BooleanVar(value=True)
        self.optical_H_text = tk.StringVar()
        self.optical_L_text = tk.StringVar()
        self.optical_D_text = tk.StringVar()

        self.build_controls()
        self.build_plot()
        self.update_optical_labels()
        self.plot()

    def build_controls(self) -> None:
        panel = tk.Frame(self.root, padx=16, pady=16)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(
            panel,
            text="Пропускание структуры с дефектом",
            font=("Arial", 14, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        tk.Label(
            panel,
            text=(
                "Структура с дефектом: (HL)^N D (LH)^N.\n"
                "Толщины задаются как h/λ0: в долях\n"
                "расчетной длины волны λ0."
            ),
            justify=tk.LEFT,
            wraplength=300,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 14))

        fields = [
            ("h_H", "Геометрическая толщина слоя H, hH/λ0", f"{DEFAULT_H_H:g}"),
            ("h_L", "Геометрическая толщина слоя L, hL/λ0", f"{DEFAULT_H_L:g}"),
            ("h_D", "Геометрическая толщина дефекта D, hD/λ0", f"{DEFAULT_H_D:g}"),
            ("n_H", "Показатель преломления nH", f"{DEFAULT_N_H:g}"),
            ("n_L", "Показатель преломления nL", f"{DEFAULT_N_L:g}"),
            ("n_D", "Показатель преломления nD", f"{DEFAULT_N_D:g}"),
            ("periods", "Число периодов с каждой стороны", "6"),
            ("lambda_min", "Диапазон λ/λ0: от", "0.82"),
            ("lambda_max", "до", "1.18"),
            ("points", "Количество точек", str(DEFAULT_POINTS)),
        ]

        for row, (key, label, default) in enumerate(fields, start=2):
            tk.Label(panel, text=label).grid(row=row, column=0, sticky="w", pady=3)
            entry = tk.Entry(panel, width=18)
            entry.insert(0, default)
            entry.grid(row=row, column=1, sticky="ew", pady=3)
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
            pady=(2, 0),
        )
        tk.Label(panel, textvariable=self.optical_D_text, fg="#444444").grid(
            row=optical_row + 2,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(2, 10),
        )

        tk.Checkbutton(
            panel,
            text="Показывать обычную стопку без дефекта",
            variable=self.show_regular,
            command=self.plot,
        ).grid(row=optical_row + 3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        tk.Button(panel, text="Построить график", command=self.plot).grid(
            row=optical_row + 4,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(10, 4),
        )
        tk.Button(panel, text="Сбросить параметры", command=self.reset).grid(
            row=optical_row + 5,
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

    def read_parameters(self) -> dict[str, float | int]:
        values: dict[str, float | int] = {
            "h_H": self.float_value("h_H"),
            "h_L": self.float_value("h_L"),
            "h_D": self.float_value("h_D"),
            "n_H": self.float_value("n_H"),
            "n_L": self.float_value("n_L"),
            "n_D": self.float_value("n_D"),
            "periods": self.int_value("periods"),
            "lambda_min": self.float_value("lambda_min"),
            "lambda_max": self.float_value("lambda_max"),
            "points": self.int_value("points"),
        }

        if values["h_H"] <= 0 or values["h_L"] <= 0 or values["h_D"] <= 0:
            raise ValueError("Геометрические толщины слоев должны быть больше нуля.")
        if values["n_H"] <= 0 or values["n_L"] <= 0 or values["n_D"] <= 0:
            raise ValueError("Показатели преломления должны быть больше нуля.")
        if values["periods"] <= 0:
            raise ValueError("N должно быть больше нуля.")
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
            n_D = self.float_value("n_D")
            h_H = self.float_value("h_H")
            h_L = self.float_value("h_L")
            h_D = self.float_value("h_D")
        except ValueError:
            self.optical_H_text.set("nH*hH/λ0 = ...")
            self.optical_L_text.set("nL*hL/λ0 = ...")
            self.optical_D_text.set("nD*hD/λ0 = ...")
            return

        self.optical_H_text.set(
            optical_label("nH*hH/λ0", optical_thickness(n_H, h_H, LAMBDA_0))
        )
        self.optical_L_text.set(
            optical_label("nL*hL/λ0", optical_thickness(n_L, h_L, LAMBDA_0))
        )
        self.optical_D_text.set(
            optical_label(
                "nD*hD/λ0",
                optical_thickness(n_D, h_D, LAMBDA_0),
                layer_type="defect",
            )
        )

    def reset(self) -> None:
        defaults = {
            "h_H": f"{DEFAULT_H_H:g}",
            "h_L": f"{DEFAULT_H_L:g}",
            "h_D": f"{DEFAULT_H_D:g}",
            "n_H": f"{DEFAULT_N_H:g}",
            "n_L": f"{DEFAULT_N_L:g}",
            "n_D": f"{DEFAULT_N_D:g}",
            "periods": "6",
            "lambda_min": "0.82",
            "lambda_max": "1.18",
            "points": str(DEFAULT_POINTS),
        }

        for key, value in defaults.items():
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, value)

        self.show_regular.set(True)
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
        n_D = float(values["n_D"])
        h_H = float(values["h_H"])
        h_L = float(values["h_L"])
        h_D = float(values["h_D"])
        periods = int(values["periods"])
        lambda_min = float(values["lambda_min"])
        lambda_max = float(values["lambda_max"])
        points = int(values["points"])

        wavelengths = np.linspace(lambda_min * LAMBDA_0, lambda_max * LAMBDA_0, points)

        self.ax.clear()
        self.update_optical_labels()

        if self.show_regular.get():
            regular_layers = bragg_stack(
                periods=2 * periods,
                n_H=n_H,
                n_L=n_L,
                h_H=h_H,
                h_L=h_L,
                first_layer="H",
            )
            T_regular = [
                reflection_transmission(regular_layers, wavelength)[1]
                for wavelength in wavelengths
            ]
            self.ax.plot(
                wavelengths / LAMBDA_0,
                T_regular,
                linewidth=1.0,
                color="#4F5660",
                label=f"без дефекта: (HL)^{2 * periods}",
            )

        defect_layers = defect_stack(
            periods=periods,
            n_H=n_H,
            n_L=n_L,
            n_D=n_D,
            h_H=h_H,
            h_L=h_L,
            h_D=h_D,
        )
        T_defect = [
            reflection_transmission(defect_layers, wavelength)[1]
            for wavelength in wavelengths
        ]

        optical_D = optical_thickness(n_D, h_D, LAMBDA_0)
        defect_label = (
            f"с дефектом: (HL)^{periods} D (LH)^{periods}, "
            f"nD*hD/λ0 = {optical_D:g}"
        )
        if is_half_wave(optical_D):
            defect_label += " — полуволновой дефект"

        self.ax.plot(
            wavelengths / LAMBDA_0,
            T_defect,
            linewidth=1.2,
            color="#6F4ACB",
            label=defect_label,
        )

        self.ax.set_title("Пропускание с дефектной пластинкой")
        self.ax.set_xlabel(r"$\lambda/\lambda_0$")
        self.ax.set_ylabel("T — коэффициент пропускания")
        self.ax.set_xlim(lambda_min, lambda_max)
        self.ax.set_ylim(0, 1.05)
        self.ax.minorticks_on()
        self.ax.grid(which="major", alpha=0.6, linewidth=1.0)
        self.ax.grid(which="minor", alpha=0.38, linewidth=0.7)
        self.ax.legend(frameon=False, loc="upper right")

        self.figure.tight_layout()
        self.canvas.draw()


def main() -> None:
    root = tk.Tk()
    root.geometry("1160x660")
    TransmissionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
