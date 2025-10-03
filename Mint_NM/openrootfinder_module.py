import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import ipywidgets as widgets

class RootFinderOpen:
    def __init__(self, f, fprime=None, x0=None, x1=None, tol=1e-6, max_iter=20):
        self.f = f
        self.fprime = fprime
        self.x0 = x0
        self.x1 = x1
        self.tol = tol
        self.max_iter = max_iter
        self.newton_guesses = self._run_newton()
        self.secant_guesses = self._run_secant()

    def _run_newton(self):
        if self.fprime is None or self.x0 is None:
            return []
        x = self.x0
        guesses = [x]
        for _ in range(self.max_iter):
            fx, fpx = self.f(x), self.fprime(x)
            if abs(fpx) < 1e-10:
                break
            x_new = x - fx / fpx
            guesses.append(x_new)
            if abs(x_new - x) < self.tol:
                break
            x = x_new
        return guesses

    def _run_secant(self):
        if self.x0 is None or self.x1 is None:
            return []
        x0, x1 = self.x0, self.x1
        guesses = [x0, x1]
        for _ in range(self.max_iter):
            f0, f1 = self.f(x0), self.f(x1)
            if abs(f1 - f0) < 1e-10:
                break
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            guesses.append(x2)
            if abs(x2 - x1) < self.tol:
                break
            x0, x1 = x1, x2
        return guesses

    def _make_animation_open(self, method='Newton', interval_ms=750):
        if method == 'Newton':
            guesses = self.newton_guesses
            title_method = "Newton's Method"
            line_label = "Tangent Line"
        else:
            guesses = self.secant_guesses
            title_method = "Secant Method"
            line_label = "Secant Line"

        if not guesses or len(guesses) < 2:
            return None

        total_steps = len(guesses)
        min_g, max_g = min(guesses), max(guesses)

        x_span = max_g - min_g if max_g > min_g else 1.0
        x_pad = max(0.2 * x_span, 1.0)
        x_min, x_max = min_g - x_pad, max_g + x_pad

        x_vals = np.linspace(x_min, x_max, 400)
        y_vals = self.f(x_vals)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_vals, y_vals, label='f(x)', color='blue')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.7)

        line, = ax.plot([], [], marker='o', linestyle='-', color='red', markersize=8, label='Iterations')
        aux_line, = ax.plot([], [], color='green', linestyle='--', label=line_label)
        current_point, = ax.plot([], [], marker='o', color='gold', markersize=10, markeredgecolor='red', label='Current $x_k$')
        vertical_line, = ax.plot([], [], color='purple', linestyle=':', linewidth=2, label='Next guess connector')

        ax.set_title(f"{title_method} Animation", fontsize=14)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("f(x)", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        ax.set_xlim(x_min, x_max)

        f_g = [self.f(g) for g in guesses]
        y_min, y_max = min(f_g), max(f_g)

        y_span = y_max - y_min if y_max > y_min else 1.0
        y_pad = max(0.2 * y_span, 1.0)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        def update_open(frame):
            step = frame
            shown_guesses = guesses[:step+1]
            f_guesses = [self.f(g) for g in shown_guesses]

            line.set_data(shown_guesses, f_guesses)
            current_point.set_data([shown_guesses[-1]], [f_guesses[-1]])
            aux_line.set_data([], [])
            vertical_line.set_data([], [])
            ax.set_title(f"{title_method} - Step {step}/{total_steps-1}", fontsize=14)

            if step > 0:
                x_k_prev = shown_guesses[step - 1]
                f_k_prev = self.f(x_k_prev)

                if method == 'Newton':
                    m = self.fprime(x_k_prev)
                elif method == 'Secant' and step > 1:
                    x_k_minus_2 = shown_guesses[step - 2]
                    f_k_minus_2 = self.f(x_k_minus_2)
                    m = (f_k_prev - f_k_minus_2) / (x_k_prev - x_k_minus_2) if abs(x_k_prev - x_k_minus_2) > 1e-10 else None
                else:
                    m = None

                if m is not None:
                    x_line = np.linspace(x_min, x_max, 400)
                    y_line = f_k_prev + m * (x_line - x_k_prev)
                    aux_line.set_data(x_line, y_line)
                    if abs(m) > 1e-12:
                        x_intersect = x_k_prev - f_k_prev / m
                        vertical_line.set_data([x_intersect, x_intersect], [0, self.f(shown_guesses[step])])

            return line, current_point, aux_line, vertical_line

        anim = FuncAnimation(
            fig, update_open,
            frames=total_steps,
            interval=interval_ms,
            blit=True,
            repeat=False
        )
        plt.close(fig)
        return HTML(anim.to_jshtml())

    def show_toggle_open(self, interval_ms=750):
        newton_anim_html = self._make_animation_open('Newton', interval_ms)
        secant_anim_html = self._make_animation_open('Secant', interval_ms)

        if newton_anim_html is None or secant_anim_html is None:
            print("Cannot create animations. Check function definition or initial points (x0, x1).")
            return

        newton_content = widgets.HTML(value=newton_anim_html.data)
        secant_content = widgets.HTML(value=secant_anim_html.data)

        tab_container = widgets.Tab()
        tab_container.children = [newton_content, secant_content]

        tab_container.set_title(0, 'Newton')
        tab_container.set_title(1, 'Secant')

        display(tab_container)
