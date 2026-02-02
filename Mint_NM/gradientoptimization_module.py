import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import ipywidgets as widgets

class OptimizerGrad:

    def __init__(self, f, grad, hess, x0, lr=0.1, tol=1e-4, max_iter=200):
        self.f = f
        self.grad = grad
        self.hess = hess
        self.x0 = np.asarray(x0, dtype=float)
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter

        self.newton_steps = self._run_newton()
        self.gd_steps = self._run_gradient_descent()

        self._define_plot_domain()


    def _run_gradient_descent(self):
        x = self.x0.copy()
        steps = [x.copy()]
        for _ in range(self.max_iter):
            g = self.grad(x)
            x_new = x - self.lr * g
            steps.append(x_new.copy())
            if np.linalg.norm(g) < self.tol:
                break
            x = x_new
        return [tuple(s) for s in steps]

    def _run_newton(self):
        x = self.x0.copy()
        steps = [x.copy()]
        for _ in range(self.max_iter):
            g = self.grad(x)
            H = self.hess(x)
            try:
                p = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            x_new = x - p
            steps.append(x_new.copy())
            if np.linalg.norm(g) < self.tol:
                break
            x = x_new
        return [tuple(s) for s in steps]


    def _define_plot_domain(self):
        all_points = np.array(self.newton_steps + self.gd_steps)
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        dx, dy = (x_max - x_min) * 0.3, (y_max - y_min) * 0.3
        x_min, x_max = x_min - dx, x_max + dx
        y_min, y_max = y_min - dy, y_max + dy
        self.x = np.linspace(x_min, x_max, 150)
        self.y = np.linspace(y_min, y_max, 150)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.Z = np.array([[self.f([xi, yi]) for xi in self.x] for yi in self.y])


    def _animate_optimization(self, steps, method_name, interval_ms=500, show_vectors=False):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.subplots_adjust(wspace=0.3)

        contour = ax1.contourf(self.X, self.Y, self.Z, levels=50, cmap='viridis')
        ax1.contour(self.X, self.Y, self.Z, levels=15, colors='k', alpha=0.3, linewidths=0.5)
        ax1.plot(self.x0[0], self.x0[1], 'r*', markersize=10, label='Initial Guess')
        path_line, = ax1.plot([], [], 'r-', lw=1.5, alpha=0.7)
        current_dot, = ax1.plot([], [], 'ro', markersize=6, label='Current Point')
        ax1.legend(loc='lower left')
        cbar = fig.colorbar(contour, ax=ax1)
        cbar.set_label("f(x, y)", rotation=270, labelpad=15)

        title_text = ax1.text(0.5, 1.03, '', transform=ax1.transAxes,
                              ha='center', fontsize=12)

        indicator_box = ax1.annotate(
            '',
            xy=(0.02, 0.98),
            xycoords='axes fraction',
            textcoords='axes fraction',
            ha='left', va='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8, ec="black")
        )

        quivers = []

        grad_vals = np.array([[self.grad([xi, yi]) for xi in self.x] for yi in self.y])
        Gx, Gy = grad_vals[:, :, 0], grad_vals[:, :, 1]
        grad_mag = np.sqrt(Gx**2 + Gy**2)

        if "Gradient" in method_name:
            Gx_u = np.divide(Gx, grad_mag, out=np.zeros_like(Gx), where=grad_mag != 0)
            Gy_u = np.divide(Gy, grad_mag, out=np.zeros_like(Gy), where=grad_mag != 0)

            skip = (slice(None, None, 6), slice(None, None, 6))
            Q = ax2.quiver(
                self.X[skip], self.Y[skip],
                Gx_u[skip], Gy_u[skip],
                grad_mag[skip], cmap='plasma',
                angles='xy', scale_units='xy', scale=15, width=0.005
            )
            ax2.set_title("Gradient Vector Field ∇f(x, y)")
            fig.colorbar(Q, ax=ax2, label="‖∇f(x, y)‖")
        else:
            grad_contour = ax2.contourf(self.X, self.Y, grad_mag, levels=40, cmap='plasma')
            fig.colorbar(grad_contour, ax=ax2, label="‖∇f(x, y)‖")
            ax2.set_title("Gradient Magnitude Field ‖∇f(x, y)‖")

        ax2.set_xlabel("x")
        ax2.set_ylabel("y")

        def update(i):
            current_path_x = [step[0] for step in steps[:i+1]]
            current_path_y = [step[1] for step in steps[:i+1]]
            path_line.set_data(current_path_x, current_path_y)

            current_x, current_y = steps[i]
            current_dot.set_data([current_x], [current_y])
            f_val = self.f([current_x, current_y])

            if show_vectors and i > 0:
                x_prev, y_prev = steps[i-1]
                dx, dy = current_x - x_prev, current_y - y_prev
                q1 = ax1.quiver(x_prev, y_prev, dx, dy, angles='xy', scale_units='xy',
                                scale=1, color='red', width=0.005)
                q2 = ax2.quiver(x_prev, y_prev, dx, dy, angles='xy', scale_units='xy',
                                scale=1, color='black', width=0.005)
                quivers.extend([q1, q2])

            title_text.set_text(
                f"{method_name} Step {i+1}/{len(steps)}\n"
                f"({current_x:.4f}, {current_y:.4f})  f={f_val:.4f}"
            )

            grad_norm = np.linalg.norm(self.grad([current_x, current_y]))
            indicator_box.set_text(f"‖∇f‖ = {grad_norm:.4e}")

            return path_line, current_dot, title_text, indicator_box, *quivers

        anim = FuncAnimation(fig, update, frames=len(steps),
                             interval=interval_ms, blit=True, repeat=False)
        plt.close(fig)
        return HTML(anim.to_jshtml())


    def show_toggle_open(self, interval_ms=500):
        gd_anim = self._animate_optimization(self.gd_steps, "Gradient Descent",
                                             interval_ms, show_vectors=True)
        newton_anim = self._animate_optimization(self.newton_steps, "Newton's Method",
                                                 interval_ms, show_vectors=True)

        gd_tab = widgets.HTML(value=gd_anim.data)
        newton_tab = widgets.HTML(value=newton_anim.data)
        tab = widgets.Tab()
        tab.children = [gd_tab, newton_tab]
        tab.set_title(0, "Gradient Descent")
        tab.set_title(1, "Newton's Method")
        display(tab)
