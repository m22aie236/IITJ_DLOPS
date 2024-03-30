# feature_2.py

import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)

# Generate random data
random_values = np.array([-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6])

# Plot graphs for ReLU, Leaky ReLU, and Tanh
plt.plot(random_values, relu(random_values), label='ReLU')
plt.plot(random_values, leaky_relu(random_values), label='Leaky ReLU')
plt.plot(random_values, tanh(random_values), label='Tanh')
plt.xlabel('x')
plt.ylabel('Activation')
plt.title('Feature-2: Activation Functions')
plt.legend()
plt.show()
