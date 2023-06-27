import matplotlib.pyplot as plt
import numpy as np

plt.figure()
loglik = np.loadtxt('experiments/outputs/Watson_454_trainlikelihoodcurve_K=2.csv')
plt.plot(loglik)
plt.savefig('experiments/analyses/Watson_454_trainlikelihoodcurve_K=2.png')
# for K in range(2,21):
    