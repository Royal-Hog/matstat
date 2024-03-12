import numpy as np
import matplotlib.pyplot as plt

sizes = [20, 100]

distributions = [
    ("Normal", lambda size: np.random.normal(0, 1, size)),
    ("Cauchy", lambda size: np.random.standard_cauchy(size)),
    ("Student", lambda size: np.random.standard_t(3, size)),
    ("Puasson", lambda size: np.random.poisson(10, size)),
    ("Uniform", lambda size: np.random.uniform(-np.sqrt(3), np.sqrt(3), size))
]

for name, func, in distributions:
    fig, graph = plt.subplots(1, 2, figsize=(12, 5))
    i=0;
    for size in sizes:
        data = func(size)
        graph[i].boxplot(data, vert=False)
        graph[i].set_title(f"{name}, n={size}")
        i=i+1
    plt.tight_layout()
    plt.show()
