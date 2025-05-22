import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution
import time
import warnings
warnings.filterwarnings('ignore') 

# Define the target function
def target_function(x):
    return x**2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)

# Generate training and test data
np.random.seed(42)
X_train = np.linspace(-10, 10, 1000).reshape(-1, 1)
y_train = target_function(X_train.flatten())
X_test = np.linspace(-10, 10, 200).reshape(-1, 1)
y_test = target_function(X_test.flatten())

# Function to train MLP with Adam optimizer and evaluate
def train_mlp_adam(hidden_layer_sizes, X_train, y_train, X_test, y_test):
    start_time = time.time()
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='relu', 
                       solver='adam', max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    training_time = time.time() - start_time
    y_pred = mlp.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mlp, mse, training_time, y_pred

# Function to flatten MLP weights for evolutionary optimization
def get_mlp_weights(mlp):
    weights = []
    for coef in mlp.coefs_:
        weights.append(coef.flatten())
    for bias in mlp.intercepts_:
        weights.append(bias.flatten())
    return np.concatenate(weights)

# Function to set MLP weights from a flattened array
def set_mlp_weights(mlp, weights):
    offset = 0
    for i, coef in enumerate(mlp.coefs_):
        size = coef.size
        mlp.coefs_[i] = weights[offset:offset+size].reshape(coef.shape)
        offset += size
    for i, bias in enumerate(mlp.intercepts_):
        size = bias.size
        mlp.intercepts_[i] = weights[offset:offset+size].reshape(bias.shape)
        offset += size

# Objective function for differential evolution
def objective_function(weights, mlp, X, y):
    set_mlp_weights(mlp, weights)
    y_pred = mlp.predict(X)
    return mean_squared_error(y, y_pred)

# Function to train MLP with Differential Evolution
def train_mlp_de(hidden_layer_sizes, X_train, y_train, X_test, y_test):
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='relu', 
                       solver='adam', max_iter=1, random_state=42)
    mlp.fit(X_train, y_train)  
    n_weights = len(get_mlp_weights(mlp))
    bounds = [(-1, 1)] * n_weights  
    
    start_time = time.time()
    result = differential_evolution(
        func=objective_function, 
        args=(mlp, X_train, y_train), 
        bounds=bounds, 
        maxiter=50, 
        popsize=15, 
        seed=42
    )
    training_time = time.time() - start_time
    set_mlp_weights(mlp, result.x)
    y_pred = mlp.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mlp, mse, training_time, y_pred

# Configurations to test
configs = [
    (10,),# 1 layer, 10 neurons
    (20,),
    (50,),      # 1 layer, 50 neurons
    (20,10),
    (20, 20)   # 2 layers, 20 neurons each
]

# Store results
results = []

# Train and evaluate models
for hidden_layers in configs:
    # Train with Adam
    mlp_adam, mse_adam, time_adam, y_pred_adam = train_mlp_adam(hidden_layers, X_train, y_train, X_test, y_test)
    # Train with Differential Evolution
    mlp_de, mse_de, time_de, y_pred_de = train_mlp_de(hidden_layers, X_train, y_train, X_test, y_test)
    
    results.append({
        'layers': hidden_layers,
        'mse_adam': mse_adam,
        'time_adam': time_adam,
        'y_pred_adam': y_pred_adam,
        'mse_de': mse_de,
        'time_de': time_de,
        'y_pred_de': y_pred_de
    })

# Plot results
plt.figure(figsize=(15, 10))
for i, result in enumerate(results):
    plt.subplot(2, 3, i+1)
    plt.plot(X_test, y_test, label='True Function', color='black')
    plt.plot(X_test, result['y_pred_adam'], label='Adam', linestyle='--')
    plt.plot(X_test, result['y_pred_de'], label='DE', linestyle='-.')
    plt.title(f"Layers: {result['layers']}\nMSE Adam: {result['mse_adam']:.2e}, DE: {result['mse_de']:.2e}")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('mlp_approximation.png')
plt.close()


# Print results
print("Configuration | MSE (Adam) | Time (Adam) | MSE (DE) | Time (DE)")
print("-" * 60)
for result in results:
    print(f"{result['layers']} | {result['mse_adam']:.2e} | {result['time_adam']:.2f}s | "
          f"{result['mse_de']:.2e} | {result['time_de']:.2f}s")