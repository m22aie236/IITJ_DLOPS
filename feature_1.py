# feature_1.py

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate random data
random_values = np.array([-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6])

# Plot sigmoid graph
plt.plot(random_values, sigmoid(random_values), label='Sigmoid')
plt.xlabel('x')
plt.ylabel('Sigmoid')
plt.title('Sigmoid Function')
plt.legend()
plt.show()
