import numpy as np

distributions = [
    ("normal", lambda size: np.random.normal(0, 1, size)),
    ("cauchy", lambda size: np.random.standard_cauchy(size)),
    ("student", lambda size: np.random.standard_t(3, size)),
    ("puasson", lambda size: np.random.poisson(10, size)),
    ("uniform", lambda size: np.random.uniform(-np.sqrt(3), np.sqrt(3), size))
]
sizes = [10, 100, 1000]
iterations = 1000
def stats(data):
    def mean(data):
        return sum(data) / len(data)
    def median(data):
        if len(data) % 2 == 1: return data[len(data) // 2]
        return 0.5 * (data[len(data) // 2] + data[len(data) // 2 + 1])
    def halfsum_extreme(data):
        return 0.5 * (data[0] + data[-1])
    def halfsum_quartiles(data):
        if len(data) % 4 == 0:
            return 0.5 * (data[len(data) // 4 - 1] + data[3 * len(data) // 4 - 1]) / 2
        return 0.5 * (data[len(data) // 4] + data[3 * len(data) // 4]) / 2
    def trimmed_mean(sample):
        r = round(len(sample) / 4)
        return sum(sample[r:len(sample) - r + 1]) / 4

    return (mean(data), median(data), halfsum_extreme(data), halfsum_quartiles(data), trimmed_mean(data))

def Mean(samples):
    transposed = [[row[i] for row in samples] for i in range(len(samples[0]))]
    return tuple([sum(transposed[i]) / len(transposed[i]) for i in range(len(transposed))])


def Dispersion(samples):
    t_means = [[row[i] for row in samples] for i in range(len(samples[0]))]
    t_squares = [[row[i] ** 2 for row in samples] for i in range(len(samples[0]))]

    def MakerFunc(data):
        return sum(data[1]) / len(data[1]) - (sum(data[0]) / len(data[0])) ** 2

    return tuple([MakerFunc(tuple([t_means[i], t_squares[i]])) for i in range(len(t_means))])


for name,func in distributions:
    print("Statistics for "+name+" distribution:\n")
    for size in sizes:
        data = []
        for i in range(iterations):
            data.append(stats(func(size)))
        mean=Mean(data)
        dispersion=Dispersion(data)
        print( f"""
    Size = {size}:
    Means:
    {' & '.join(['{:.2f}'.format(i) for i in mean])} 
    Dispersion:
    {' & '.join(['{:.2f}'.format(i) for i in dispersion])}

    """)
