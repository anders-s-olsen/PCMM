# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert
from src.DMM_EM import ACGEM, mixture_EM_loop

# %%
# make two oscillating signals. In the first quarter of the time series they should be in phase, in the second quarter they should be slightly out-of-phase where signal1 precedes signal2, in the fourt quarter they should be in anti-phase, and in the fourth quarter they should be slightly out-of-phase where signal2 precedes signal1.
n = 1000
t = np.linspace(0, 5, n)
noises = np.zeros((2,n))
noises[0] = np.random.normal(0, 0.05, n)
noises[1] = np.random.normal(0, 0.05, n)
signal1 = np.zeros((2,n))
signal1[0] = np.sin(2*np.pi*t) 
signal1[1] = np.sin(2*np.pi*t)-2
signal2 = np.zeros((2,n))
signal2[0] = np.sin(2*np.pi*t)
signal2[1] = np.sin(2*np.pi*t + np.pi/4)-2
signal3 = np.zeros((2,n))
signal3[0] = np.sin(2*np.pi*t)
signal3[1] = np.sin(2*np.pi*t + np.pi/2)-2
signal4 = np.zeros((2,n))
signal4[0] = np.sin(2*np.pi*t)
signal4[1] = np.sin(2*np.pi*t + 3*np.pi/4)-2
signal5 = np.zeros((2,n))
signal5[0] = np.sin(2*np.pi*t)
signal5[1] = np.sin(2*np.pi*t + np.pi)-2
signals = [signal1, signal2, signal3, signal4, signal5]

# plot the signals concatenated
fig = plt.figure(figsize=(12, 2))
gs = fig.add_gridspec(1, 5,wspace=0)
axs = gs.subplots(sharex=True, sharey=True)
for i in range(5):
    axs[i].plot(signals[i].T+noises.T)
    axs[i].set_xticks([])
    axs[i].set_yticks([])
axs[0].set_title('In phase')
axs[1].set_title('Slightly out-of-phase')
axs[2].set_title('Orthogonal')
axs[3].set_title('Slightly out-of-phase')
axs[4].set_title('Antiphase')
# plt.show()

# %%
#plot a unit circle
fig, axs = plt.subplots(1,5, figsize=(12, 2),layout='tight')
for i in range(5):
    axs[i].set_aspect('equal')
    axs[i].add_artist(plt.Circle((0, 0), 1, fill=False))
    axs[i].set_xlim(-1.5, 1.5)
    axs[i].set_ylim(-1.5, 1.5)
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].vlines(0, -1.5, 1.5, color='black', linewidth=0.5)
    axs[i].hlines(0, -1.5, 1.5, color='black', linewidth=0.5)
    axs[i].axis('off')
    sig1 = signals[i][0]+noises[0]
    sig2 = signals[i][1]+noises[1]
    hil1 = np.angle(hilbert(sig1-np.mean(sig1)))
    hil2 = np.angle(hilbert(sig2-np.mean(sig2)))
    inp1 = (np.cos(hil1)+1j*np.sin(hil1))/np.sqrt(2)
    inp2 = (np.cos(hil2)+1j*np.sin(hil2))/np.sqrt(2)
    hils = np.array([inp1, inp2])
    model = ACGEM.ACG(p=2,complex=True)
    out = mixture_EM_loop.mixture_EM_loop(model,data=hils.T,init='diametrical_clustering')
    fit = out[0]['Lambda'][0,0,1]
    circular_difference_signal = hil1-hil2
    axs[i].plot(np.cos(circular_difference_signal), np.sin(circular_difference_signal), 'r.', alpha=0.5, linewidth=2.5)
    axs[i].plot(np.real(fit), np.imag(fit), 'b.', alpha=0.5, linewidth=2.5)

# %%
out[0]['Lambda'][0]


