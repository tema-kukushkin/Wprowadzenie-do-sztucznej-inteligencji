import autograd.numpy as np
from autograd import grad
import time
import matplotlib.pyplot as plt
from typing import Callable, Sequence, NamedTuple

class SolverParameters(NamedTuple):
    learning_rate: float
    max_iterations: int 
    tolerance: float  

def solver(
    eval_func: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    params: SolverParameters
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
            break  # If the gradient is small, than stop.
        
        x -= params.learning_rate * grad_value  # take a step of gradient descent
    
    execution_time = time.time() - start_time
    return x, history, execution_time
def quadratic_function(x):
    return np.sum(x**2)

def rosenbrock_function(x):
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1))

def ackley_function(x):
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    return -a * np.exp(-b * np.sqrt(np.sum(x**2) / d)) - np.exp(np.sum(np.cos(c*x)) / d) + a + np.exp(1)

# Method parameters
params = SolverParameters(learning_rate=0.0001, max_iterations=10000, tolerance=1e-6)  
x0 = [-1.2, 1.0]

#Run gradient descent for each function
for func, name in zip([quadratic_function, rosenbrock_function, ackley_function],
                       ["Quadratic", "Rosenbrock", "Ackley"]):
    x_opt, history, exec_time = solver(func, x0, params)
    print(f"{name} function: Optimal x = {x_opt}, Execution time = {exec_time:.4f}s")
    
    plt.plot([t for t, v in history], [v for t, v in history], label=name)
    
plt.xlabel("Iteration")
plt.ylabel("Function value")
plt.legend()
plt.title("Gradient Descent Convergence")
plt.show()
