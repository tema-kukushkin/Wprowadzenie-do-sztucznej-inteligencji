import autograd.numpy as np
from autograd import grad
import time
from typing import Callable, Sequence
from dataclasses import dataclass


@dataclass
class SolverParameters:
    learning_rate: float
    max_iterations: int
    tolerance: float


def solver(
    eval_func: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    params: SolverParameters,
):
    x = np.array(x0, dtype=float)
    gradient = grad(eval_func)
    history = []
    start_time = time.time()

    for t in range(params.max_iterations):
        grad_value = gradient(x)
        grad_value = np.clip(grad_value, -10, 10)
        history.append((t, eval_func(x)))

        if np.linalg.norm(grad_value) < params.tolerance:
            break  # If the gradient is small, stop.

        x -= params.learning_rate * grad_value  # Gradient descent step

    execution_time = time.time() - start_time
    return x, history, execution_time
