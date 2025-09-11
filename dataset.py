import numpy as np 

def generate_data(num_samples=100):

    x = np.linspace(-2*np.pi, 2*np.pi, num_samples).reshape(-1, 1)

    y = np.sin(x) + 0.1 * np.random.randn(num_samples, 1)

    return x, y
