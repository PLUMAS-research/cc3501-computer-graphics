import numpy as np
import matplotlib.pyplot as plt
import click


# %% Métodos de integración

def euler_step(f, t, y, h):
    """Un paso del método de Euler explícito."""
    return y + h * f(t, y)


def rk4_step(f, t, y, h):
    """Un paso del método de Runge-Kutta de 4to orden."""
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h * k1 / 2)
    k3 = f(t + h / 2, y + h * k2 / 2)
    k4 = f(t + h, y + h * k3)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def integrate(step_func, f, y0, t0, tf, h):
    """Integra una EDO con el método dado por step_func.

    Args:
        step_func: función (f, t, y, h) -> y_next
        f: función (t, y) -> dy/dt
        y0: condición inicial (escalar o array)
        t0, tf: intervalo de tiempo
        h: paso temporal

    Returns:
        t: array de tiempos
        y: array de estados
    """
    n_steps = int((tf - t0) / h)
    t = np.zeros(n_steps + 1)
    y0 = np.asarray(y0, dtype=np.float64)
    y = np.zeros((n_steps + 1,) + y0.shape)

    t[0] = t0
    y[0] = y0

    for i in range(n_steps):
        y[i + 1] = step_func(f, t[i], y[i], h)
        t[i + 1] = t[i] + h

    return t, y


# %% Definición de las EDOs

def cubic_ode(t, y):
    """dy/dt = y^3 - y"""
    return y**3 - y


def circular_ode(t, state):
    """dx/dt = -y, dy/dt = x (movimiento circular)."""
    x, y = state
    return np.array([-y, x])


def circular_exact(t, x0, y0):
    """Solución analítica del sistema circular."""
    x = x0 * np.cos(t) - y0 * np.sin(t)
    y = x0 * np.sin(t) + y0 * np.cos(t)
    return np.column_stack((x, y))


def vanderpol_ode(t, state, mu=2.0):
    """Oscilador de van der Pol: y'' - mu*(1-y^2)*y' + y = 0"""
    y, v = state
    return np.array([v, mu * (1 - y**2) * v - y])


# %% Funciones de visualización individuales

def show_cubic():
    """EDO cúbica: dy/dt = y^3 - y"""
    y0 = 0.5
    t0, tf = 0.0, 2.0
    h = 0.05
    h_ref = 0.001

    print(f"  EDO: dy/dt = y^3 - y")
    print(f"  y(0) = {y0}, h = {h}")

    t_euler, y_euler = integrate(euler_step, cubic_ode, y0, t0, tf, h)
    t_rk4, y_rk4 = integrate(rk4_step, cubic_ode, y0, t0, tf, h)
    t_ref, y_ref = integrate(rk4_step, cubic_ode, y0, t0, tf, h_ref)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_ref, y_ref, "k-", linewidth=2, label="Referencia (RK4 h=0.001)")
    ax.plot(t_euler, y_euler, "r-", linewidth=1, label=f"Euler (h={h})")
    ax.plot(t_rk4, y_rk4, "b-", linewidth=1, label=f"RK4 (h={h})")
    ax.set_xlabel("t")
    ax.set_ylabel("y(t)")
    ax.set_title(r"EDO cúbica: $\frac{dy}{dt} = y^3 - y$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()


def show_spiral():
    """Sistema circular: dx/dt = -y, dy/dt = x"""
    state0 = np.array([1.0, 0.0])
    t0, tf = 0.0, 10.0
    h = 0.2

    print(f"  Sistema: dx/dt = -y, dy/dt = x")
    print(f"  (x0, y0) = ({state0[0]}, {state0[1]}), h = {h}")

    t_euler, y_euler = integrate(euler_step, circular_ode, state0, t0, tf, h)
    t_rk4, y_rk4 = integrate(rk4_step, circular_ode, state0, t0, tf, h)
    t_exact = np.arange(t0, tf + 0.01, 0.01)
    y_exact = circular_exact(t_exact, state0[0], state0[1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Trayectoria en el plano x-y
    ax1.plot(y_exact[:, 0], y_exact[:, 1], "k-", linewidth=2, label="Exacta")
    ax1.plot(y_euler[:, 0], y_euler[:, 1], "r-", linewidth=1, label=f"Euler (h={h})")
    ax1.plot(y_rk4[:, 0], y_rk4[:, 1], "b-", linewidth=1, label=f"RK4 (h={h})")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Trayectoria en el plano")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    # Evolución del radio
    r_exact = np.sqrt(y_exact[:, 0] ** 2 + y_exact[:, 1] ** 2)
    r_euler = np.sqrt(y_euler[:, 0] ** 2 + y_euler[:, 1] ** 2)
    r_rk4 = np.sqrt(y_rk4[:, 0] ** 2 + y_rk4[:, 1] ** 2)

    ax2.plot(t_exact, r_exact, "k-", linewidth=2, label="Exacta")
    ax2.plot(t_euler, r_euler, "r-", linewidth=1, label=f"Euler (h={h})")
    ax2.plot(t_rk4, r_rk4, "b-", linewidth=1, label=f"RK4 (h={h})")
    ax2.set_xlabel("t")
    ax2.set_ylabel("radio")
    ax2.set_title("Evolución del radio")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.show()


def show_vanderpol():
    """Oscilador de van der Pol"""
    state0 = np.array([0.5, 0.0])
    t0, tf = 0.0, 20.0
    h = 0.1
    h_ref = 0.01

    print(f"  Oscilador de van der Pol (mu=2)")
    print(f"  (y0, v0) = ({state0[0]}, {state0[1]}), h = {h}")

    f = lambda t, y: vanderpol_ode(t, y)

    t_euler, y_euler = integrate(euler_step, f, state0, t0, tf, h)
    t_rk4, y_rk4 = integrate(rk4_step, f, state0, t0, tf, h)
    t_ref, y_ref = integrate(rk4_step, f, state0, t0, tf, h_ref)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # y(t)
    ax1.plot(t_ref, y_ref[:, 0], "k-", linewidth=2, label="Referencia (RK4 h=0.01)")
    ax1.plot(t_euler, y_euler[:, 0], "r-", linewidth=1, label=f"Euler (h={h})")
    ax1.plot(t_rk4, y_rk4[:, 0], "b-", linewidth=1, label=f"RK4 (h={h})")
    ax1.set_xlabel("t")
    ax1.set_ylabel("y(t)")
    ax1.set_title(r"Van der Pol: $y'' - \mu(1-y^2)y' + y = 0$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Espacio de fases
    ax2.plot(y_ref[:, 0], y_ref[:, 1], "k-", linewidth=2, label="Referencia")
    ax2.plot(y_euler[:, 0], y_euler[:, 1], "r-", linewidth=1, alpha=0.7, label=f"Euler (h={h})")
    ax2.plot(y_rk4[:, 0], y_rk4[:, 1], "b-", linewidth=1, alpha=0.7, label=f"RK4 (h={h})")
    ax2.set_xlabel("y")
    ax2.set_ylabel("y'")
    ax2.set_title("Espacio de fases")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.show()


def show_convergence():
    """Comparación de convergencia: error vs. tamaño de paso"""
    y0 = np.array([1.0, 0.0])
    t0, tf = 0.0, 2 * np.pi  # una vuelta completa
    y_exact_final = circular_exact(np.array([tf]), y0[0], y0[1])[0]

    steps = [0.5, 0.25, 0.1, 0.05, 0.025, 0.01]

    print("  Comparación de convergencia en el sistema circular")
    print(f"  Intervalo: [0, 2*pi], solución exacta final: ({y_exact_final[0]:.4f}, {y_exact_final[1]:.4f})")
    print(f"  Pasos evaluados: {steps}")

    errors_euler = []
    errors_rk4 = []

    for h in steps:
        _, y_e = integrate(euler_step, circular_ode, y0, t0, tf, h)
        _, y_r = integrate(rk4_step, circular_ode, y0, t0, tf, h)
        errors_euler.append(np.linalg.norm(y_e[-1] - y_exact_final))
        errors_rk4.append(np.linalg.norm(y_r[-1] - y_exact_final))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(steps, errors_euler, "ro-", linewidth=2, markersize=6, label="Euler (orden 1)")
    ax.loglog(steps, errors_rk4, "bo-", linewidth=2, markersize=6, label="RK4 (orden 4)")

    # Líneas de referencia para los órdenes esperados
    h_arr = np.array(steps)
    ax.loglog(h_arr, 0.5 * h_arr, "r--", alpha=0.4, label=r"$\propto h^1$")
    ax.loglog(h_arr, 0.5 * h_arr**4, "b--", alpha=0.4, label=r"$\propto h^4$")

    ax.set_xlabel("Tamaño de paso h")
    ax.set_ylabel("Error (norma L2 en t=2π)")
    ax.set_title("Convergencia: error global vs. tamaño de paso")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.show()


def save_summary(output):
    """Genera un PNG con los cuatro gráficos."""
    print(f"  Generando {output}...")

    # Datos para los tres casos
    # Caso 1: cúbica
    y0_1 = 0.5
    h1, h_ref1 = 0.05, 0.001
    t_e1, y_e1 = integrate(euler_step, cubic_ode, y0_1, 0, 2, h1)
    t_r1, y_r1 = integrate(rk4_step, cubic_ode, y0_1, 0, 2, h1)
    t_ref1, y_ref1 = integrate(rk4_step, cubic_ode, y0_1, 0, 2, h_ref1)

    # Caso 2: circular
    state0 = np.array([1.0, 0.0])
    h2 = 0.2
    t_e2, y_e2 = integrate(euler_step, circular_ode, state0, 0, 10, h2)
    t_r2, y_r2 = integrate(rk4_step, circular_ode, state0, 0, 10, h2)
    t_ex2 = np.arange(0, 10.01, 0.01)
    y_ex2 = circular_exact(t_ex2, 1.0, 0.0)

    # Caso 3: van der Pol
    state0_vdp = np.array([0.5, 0.0])
    h3, h_ref3 = 0.1, 0.01
    f_vdp = lambda t, y: vanderpol_ode(t, y)
    t_e3, y_e3 = integrate(euler_step, f_vdp, state0_vdp, 0, 20, h3)
    t_r3, y_r3 = integrate(rk4_step, f_vdp, state0_vdp, 0, 20, h3)
    t_ref3, y_ref3 = integrate(rk4_step, f_vdp, state0_vdp, 0, 20, h_ref3)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Cúbica
    ax = axes[0, 0]
    ax.plot(t_ref1, y_ref1, "k-", linewidth=2, label="Referencia")
    ax.plot(t_e1, y_e1, "r-", linewidth=1, label=f"Euler (h={h1})")
    ax.plot(t_r1, y_r1, "b-", linewidth=1, label=f"RK4 (h={h1})")
    ax.set_xlabel("t")
    ax.set_ylabel("y(t)")
    ax.set_title(r"$\frac{dy}{dt} = y^3 - y$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Circular: trayectoria
    ax = axes[0, 1]
    ax.plot(y_ex2[:, 0], y_ex2[:, 1], "k-", linewidth=2, label="Exacta")
    ax.plot(y_e2[:, 0], y_e2[:, 1], "r-", linewidth=1, label=f"Euler (h={h2})")
    ax.plot(y_r2[:, 0], y_r2[:, 1], "b-", linewidth=1, label=f"RK4 (h={h2})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"$\frac{dx}{dt} = -y, \frac{dy}{dt} = x$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Circular: radio
    ax = axes[1, 0]
    ax.plot(t_ex2, np.sqrt(y_ex2[:, 0] ** 2 + y_ex2[:, 1] ** 2), "k-", linewidth=2, label="Exacta")
    ax.plot(t_e2, np.sqrt(y_e2[:, 0] ** 2 + y_e2[:, 1] ** 2), "r-", linewidth=1, label=f"Euler (h={h2})")
    ax.plot(t_r2, np.sqrt(y_r2[:, 0] ** 2 + y_r2[:, 1] ** 2), "b-", linewidth=1, label=f"RK4 (h={h2})")
    ax.set_xlabel("t")
    ax.set_ylabel("radio")
    ax.set_title("Evolución del radio (sistema circular)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Van der Pol
    ax = axes[1, 1]
    ax.plot(t_ref3, y_ref3[:, 0], "k-", linewidth=2, label="Referencia")
    ax.plot(t_e3, y_e3[:, 0], "r-", linewidth=1, label=f"Euler (h={h3})")
    ax.plot(t_r3, y_r3[:, 0], "b-", linewidth=1, label=f"RK4 (h={h3})")
    ax.set_xlabel("t")
    ax.set_ylabel("y(t)")
    ax.set_title(r"Van der Pol: $y'' - \mu(1-y^2)y' + y = 0$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"  Guardado en {output}")
    plt.close(fig)


# %% Comando click

@click.command("edo_case_studies", short_help="Casos de estudio de integración de EDO")
@click.option("--save", type=str, default=None, help="Guardar resumen como PNG en la ruta indicada")
def edo_case_studies(save):
    """Casos de estudio de integración numérica de EDO.

    Compara los métodos de Euler y RK4 en tres problemas:
    EDO cúbica, movimiento circular, y oscilador de van der Pol.
    """
    if save:
        save_summary(save)
        return

    options = [
        ("EDO cúbica: dy/dt = y^3 - y", show_cubic),
        ("Movimiento circular: dx/dt = -y, dy/dt = x", show_spiral),
        ("Oscilador de van der Pol", show_vanderpol),
        ("Convergencia: error vs. tamaño de paso", show_convergence),
        ("Salir", None),
    ]

    while True:
        print("\n=== Casos de estudio: integración de EDO ===\n")
        for i, (label, _) in enumerate(options, 1):
            print(f"  {i}. {label}")

        choice = input("\nElige una opción: ")

        try:
            idx = int(choice) - 1
        except ValueError:
            print("Opción no válida.")
            continue

        if idx < 0 or idx >= len(options):
            print("Opción no válida.")
            continue

        if options[idx][1] is None:
            print("\n¡Chao!")
            break

        print()
        options[idx][1]()
