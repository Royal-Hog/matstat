import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
sizes = [10, 50, 1000]
discr = {10: 8, 50: 16, 1000: 32}

distributions = [
    ("Normal", lambda size: np.random.normal(0, 1, size)),
    ("Cauchy", lambda size: np.random.standard_cauchy(size)),
    ("Student", lambda size: np.random.standard_t(3, size)),
    ("Puasson", lambda size: np.random.poisson(10, size)),
    ("Uniform", lambda size: np.random.uniform(-np.sqrt(3), np.sqrt(3), size))
]

for name, func in distributions:
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(name)
    i=0
    for size in sizes:
        data = func(size)
        graph = plt.subplot2grid((3, 2), (i // 2, i % 2))
        graph.hist(data, bins=discr[size], color='green', density=True,)
        x = np.linspace(min(data), max(data), 1000)
        if name == "Uniform":
            y = stats.uniform.pdf(x, -np.sqrt(3), 2 * np.sqrt(3))
        else:
            y = stats.gaussian_kde(data).evaluate(x)
        graph.plot(x, y, color='red',)
        graph.set_title(f"n={size}")
        i=i+1

    plt.tight_layout()
    plt.show()
