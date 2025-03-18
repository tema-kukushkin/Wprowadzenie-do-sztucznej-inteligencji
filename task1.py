import autograd.numpy as np
import matplotlib.pyplot as plt
from solver import solver, SolverParameters


def quadratic_function(x):
    return np.sum(x**2)


def rosenbrock_function(x):
    return sum(
        100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1)
    )


def ackley_function(x):
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    return (
        -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
        - np.exp(np.sum(np.cos(c * x)) / d)
        + a
        + np.exp(1)
    )


# Method parameters
params = SolverParameters(learning_rate=0.001, max_iterations=10000, tolerance=1e-6)

D = 10
x0 = np.random.uniform(2, 2, D)

# Run gradient descent for each function
for func, name in zip(
    [quadratic_function, rosenbrock_function, ackley_function],
    ["Quadratic", "Rosenbrock", "Ackley"],
):
    x_opt, history, exec_time = solver(func, x0, params)
    print(f"{name} function: Optimal x = {x_opt}, Execution time = {exec_time:.4f}s")

    plt.plot([t for t, v in history], [v for t, v in history], label=name)

plt.xlabel("Iteration")
plt.ylabel("Function value")
plt.legend()
plt.title("Gradient Descent Convergence")
plt.show()
