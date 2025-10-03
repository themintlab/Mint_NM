import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import ipywidgets as widgets

class RootFinderClosed:
    def __init__(self, f, a=None, b=None, tol=1e-6, max_iter=20):
        self.f = f
        self.a = a
        self.b = b
        self.tol = tol
        self.max_iter = max_iter
        self.bisect_intervals, self.bisect_guesses = self._run_bisection()
        self.falsepos_intervals, self.falsepos_guesses = self._run_false_position()

    def _run_bisection(self):
        if self.a is None or self.b is None:
            return [], []
        a, b = self.a, self.b
        if self.f(a) * self.f(b) > 0:
            return [], []

        intervals = []
        guesses = []
        for _ in range(self.max_iter):
            c = (a + b) / 2.0
            intervals.append((a, b, c))
            guesses.append(c)
            if abs(self.f(c)) < self.tol or abs(b - a) < self.tol:
                break
            if self.f(a) * self.f(c) < 0:
                b = c
            else:
                a = c
        return intervals, guesses

    def _run_false_position(self):
        if self.a is None or self.b is None:
            return [], []
        a, b = self.a, self.b
        if self.f(a) * self.f(b) > 0:
            return [], []

        intervals = []
        guesses = []
        for _ in range(self.max_iter):
            fa, fb = self.f(a), self.f(b)
            if abs(fb - fa) < 1e-12:
                break
            c = b - fb * (b - a) / (fb - fa)
            intervals.append((a, b, c))
            guesses.append(c)
            if abs(self.f(c)) < self.tol:
                break
            if fa * self.f(c) < 0:
                b = c
                fb = self.f(b)
            else:
                a = c
                fa = self.f(a)
        return intervals, guesses

    def _make_animation_closed(self, method='Bisection', interval_ms=750):
        if method == 'Bisection':
            intervals, guesses = self.bisect_intervals, self.bisect_guesses
            title_method = "Bisection Method"
            guess_label = "Midpoint"
        else:
            intervals, guesses = self.falsepos_intervals, self.falsepos_guesses
            title_method = "False Position Method"
            guess_label = "Point c"

        if not guesses:
            return None

        total_steps = len(guesses)
        all_points = [g for g in guesses] + [p for ab in intervals for p in ab[:2]]
        min_g, max_g = min(all_points), max(all_points)

        x_span = max_g - min_g if max_g > min_g else 1.0
        x_pad = max(0.2 * x_span, 1.0)
        x_min, x_max = min_g - x_pad, max_g + x_pad

        x_vals = np.linspace(x_min, x_max, 400)
        y_vals = self.f(x_vals)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_vals, y_vals, label='f(x)', color='blue')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.7)

        line, = ax.plot([], [], marker='o', linestyle='-', color='red', markersize=6, alpha=0.6, label='Previous guesses')
        a_line = ax.axvline(0, color='red', linestyle=':', lw=1.5, alpha=0.9, label='a (left endpoint)')
        b_line = ax.axvline(0, color='green', linestyle=':', lw=1.5, alpha=0.9, label='b (right endpoint)')
        current_point, = ax.plot([], [], marker='o', color='gold', markersize=10, markeredgecolor='black', label=guess_label)
        vertical_line, = ax.plot([], [], color='purple', linestyle=':', linewidth=2, label='Connector')

        ax.set_title(f"{title_method} Animation", fontsize=14)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("f(x)", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        ax.set_xlim(x_min, x_max)

        f_g = [self.f(g) for g in guesses]
        y_min, y_max = min(f_g + [0]), max(f_g + [0])
        y_span = y_max - y_min if y_max > y_min else 1.0
        y_pad = max(0.2 * y_span, 1.0)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        def update_closed(frame):
            step = frame
            shown_intervals = intervals[:step+1]
            shown_guesses = guesses[:step+1]
            f_guesses = [self.f(g) for g in shown_guesses]

            a, b, c = shown_intervals[-1]
            a_line.set_xdata([a])
            b_line.set_xdata([b])

            line.set_data(shown_guesses[:-1], f_guesses[:-1])
            current_point.set_data([c], [self.f(c)]) 
            vertical_line.set_data([c, c], [0, self.f(c)])

            ax.set_title(f"{title_method} - Step {step}/{total_steps-1}", fontsize=14)
            return line, current_point, vertical_line, a_line, b_line

        anim = FuncAnimation(
            fig, update_closed,
            frames=total_steps,
            interval=interval_ms,
            blit=True,
            repeat=False
        )
        plt.close(fig)
        return HTML(anim.to_jshtml())

    def show_toggle_closed(self, interval_ms=750):
        bisect_anim_html = self._make_animation_closed('Bisection', interval_ms)
        falsepos_anim_html = self._make_animation_closed('False Position', interval_ms)

        if bisect_anim_html is None or falsepos_anim_html is None:
            print("Cannot create animations. Check initial interval [a, b].")
            return

        bisect_content = widgets.HTML(value=bisect_anim_html.data)
        falsepos_content = widgets.HTML(value=falsepos_anim_html.data)

        tab_container = widgets.Tab()
        tab_container.children = [bisect_content, falsepos_content]

        tab_container.set_title(0, 'Bisection')
        tab_container.set_title(1, 'False Position')

        display(tab_container)
