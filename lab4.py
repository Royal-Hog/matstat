import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts

sizes = [20, 100]
discr = {20: 10, 100: 32}
distributions = {
    "Normal": (np.random.normal, {"loc": 0, "scale": 1}),
    "Random": (np.random.random, {}),
}

class Sample:

    def __init__(self, dist_func, size, **dist_params):
        self.sample = dist_func(size=size, **dist_params)


    def normal_mu_bounds(self, sgnfnce):
        size = len(self.sample)
        mean, std = np.mean(self.sample), np.std(self.sample)
        quantile = sts.t.ppf(1 - sgnfnce / 2, size - 1)
        return mean - std * quantile / np.sqrt(size - 1), mean + std * quantile / np.sqrt(size - 1)
    def normal_sigma_bounds(self, sgnfnce):
        size = len(self.sample)
        std = np.std(self.sample)
        return  (std * np.sqrt(size) )/ np.sqrt(sts.chi2.ppf(1-sgnfnce / 2, size - 1)), \
        (std * np.sqrt(size)) / np.sqrt(sts.chi2.ppf(sgnfnce / 2, size - 1))

    def random_mu_bounds(self, sgnfnce):
        size = len(self.sample)
        mean=np.mean(self.sample)
        std = np.std(self.sample)
        quantile = sts.norm.ppf(1-sgnfnce/2)

        return mean - (std * quantile / np.sqrt(size - 1)), mean + std * quantile / np.sqrt(size - 1)

    def random_sigma_bounds(self, sgnfnce):
        std= np.std(self.sample)
        size = len(self.sample)

        excess = ( np.sum((self.sample - np.mean(self.sample))**4) / size) / std**4 - 3
        return std * (1-0.5*sts.norm.ppf(1-sgnfnce/2) * np.sqrt((excess + 2) / size)), std * (1+0.5*sts.norm.ppf(1-sgnfnce/2) * np.sqrt((excess + 2) / size))

sgms_normal, sgms_random = [], []

def Informing(size):
    sgnfnce = 0.9
    my_sample = Sample(dist_func, size, **dist_params)

    if dist_name == "Normal":
        mu_min, mu_max = my_sample.normal_mu_bounds(sgnfnce)
        sgm_min, sgm_max = my_sample.normal_sigma_bounds(sgnfnce)
        sgms_normal.append((sgm_min, sgm_max))
    if dist_name == "Random":
        mu_min, mu_max = my_sample.random_mu_bounds(sgnfnce)
        sgm_min, sgm_max = my_sample.random_sigma_bounds(sgnfnce)
        sgms_random.append((sgm_min, sgm_max))

    print(f"""{dist_name} Distribution: size = {size}
    {mu_min} < Mu < {mu_max}
    {sgm_min} < Sigma < {sgm_max}\n 
""")
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(f"{dist_name}. N = {size}")
    labels = [
        'Histogram', r'$\mu_{\min}$', r'$\mu_{\max}$',
        r'$\mu_{\min} - \sigma_{\max}$', r'$\mu_{\max} + \sigma_{\max}$'
    ]
    plt.hist(my_sample.sample, bins=discr[size], density=True,color='green')
    color_table = {0: 'r', 1: 'b'}
    params = [mu_min, mu_max, mu_min - sgm_max, mu_max + sgm_max]
    for i, param in enumerate(params):
        plt.axvline(param, color=color_table[i // 2], linestyle='-', linewidth=3,marker='o')
    plt.show()


def PlotSSO(sgms):
    plt.hlines(0.5, xmin=sgms[0][0], xmax=sgms[0][1], color='b', linestyles='-')
    plt.hlines(0.7, xmin=sgms[1][0], xmax=sgms[1][1], color='r', linestyles='-')
    plt.plot([sgms[0][0], sgms[0][1]], [0.5, 0.5], 'ro', markersize=5)
    plt.plot([sgms[1][0], sgms[1][1]], [0.7, 0.7], 'ro', markersize=5)
    plt.legend(labels=[r'$N=20: [\sigma_{\min}; \sigma_{\max}]$',
                       r'$N=100: [\sigma_{\min}; \sigma_{\max}]$'
                       ])
    plt.show()

for dist_name, (dist_func, dist_params) in distributions.items():
    for i, size in enumerate(sizes):
        Informing(size)
PlotSSO(sgms_normal)
PlotSSO(sgms_random)
