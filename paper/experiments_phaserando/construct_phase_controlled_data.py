import numpy as np
import nibabel as nib
from scipy.signal import butter, filtfilt, hilbert
import h5py
subjects = np.loadtxt('paper/data/255unrelatedsubjectsIDs.txt', dtype='str')
p = 116
K = 5
T = 1200
num_subs = 2

def butter_bandpass(lowcut, highcut, fs, order=5):
    # nyq = fs
    # nyq = 0.5 * fs
    # low = lowcut / nyq
    # high = highcut / nyq
    low = lowcut
    high = highcut
    b, a = butter(order, [low, high], btype="bandpass", fs=fs)
    return b, a

def butter_bandpass_filter(data, lowcut=0.03, highcut=0.07, fs=1 / 0.720, order=5):
# def butter_bandpass_filter(data, lowcut=0.009, highcut=0.08, fs=1 / 0.720, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

atlas = nib.load('paper/data/external/Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1.dlabel.nii')
atlas_data = atlas.get_fdata()
atlas_data = np.round(atlas_data)
atlas_data = atlas_data.astype(int)[0]
filtered_data_all = []

for sub in subjects[:num_subs]:
    for session in [1,2]:
        img = nib.load('paper/data/raw/'+sub+'/fMRI/rfMRI_REST'+str(session)+'_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii')
        data = img.get_fdata()
        data = data - np.mean(data, axis=0)

        GS = np.mean(data, axis=1)
        data = data - GS[:,None]*(data.T@GS).T/(GS.T@GS)
        # parcellate data in atlas
        parcellated_data = np.zeros((data.shape[0], atlas_data.max()))
        for i in range(1, atlas_data.max()+1):
            parcellated_data[:,i-1] = np.mean(data[:,atlas_data == i], axis=1)
            # parcellated_data[:,i-1] = parcellated_data[:,i-1] - np.mean(parcellated_data[:,i-1])
        # parcellated_data = parcellated_data - np.mean(parcellated_data, axis=0)
        # filter data
        filtered_data = np.zeros(parcellated_data.shape)
        for i in range(p):
            filtered_data[:,i] = butter_bandpass_filter(parcellated_data[:,i])
        filtered_data_all.append(filtered_data)

num_repeats = 1 #repeats of clustering pattern state1..state2.. etc

times = np.linspace(0, 1200, K*num_repeats+1, dtype=int)
num_samples = times[1]-times[0]
# num_add = num_samples//10
num_add = 10
TR = 0.720
f = np.fft.fftfreq(num_samples+num_add*2, TR)
# find indices of frequencies above or equal to 0.008Hz and below or equal to 0.09Hz
positive_frequency_content_indices = np.where((f>=0))[0]
negative_frequency_content_indices = np.where((f<0))[0]
mid_frequency = np.argmin(np.abs(f-0.05))
num_phases = len(positive_frequency_content_indices)

nodes = np.arange(0, p, p//K)
# nodes = [0, 23, 46, 69, 92]
nodes = np.tile(nodes, num_repeats)
specs = []
for window in range(K*num_repeats):
    ref = filtered_data_all[0][times[window]:times[window+1],nodes[window]]
    if num_add>0:
        ref2 = np.concatenate((2*ref[0]-np.flip(ref[1:num_add+1]),ref,2*ref[-1]-np.flip(ref[-num_add-1:-1])))
        ref3 = np.concatenate((-np.flip(ref[1:num_add+1]),ref,-np.flip(ref[-num_add-1:-1])))
    specs.append(np.fft.fft(ref))

import matplotlib.pyplot as plt
plt.figure()
time = np.arange(times[0], times[-1]) * TR
plt.plot(np.concatenate([np.zeros(num_add),ref,np.zeros(num_add)])+10)
plt.plot(ref2+5)
plt.plot(ref3)
plt.savefig('tmp.png')


phase_reset_data_all = []
divide_factor = 1
phase_shifts = np.random.rand(p)*2*np.pi/divide_factor-np.pi/divide_factor #within-cluster phase shifts equal across subjects
# phase_shifts = np.zeros(p)
for scan in range(len(filtered_data_all)):
    newdata = np.zeros((T,p))
    for window in range(K*num_repeats):
        for i in range(p):
            signal = filtered_data_all[scan][times[window]:times[window+1],i]
            if num_add>0:
                signal = np.concatenate((2*signal[0]-np.flip(signal[1:num_add+1]),signal,2*signal[-1]-np.flip(signal[-num_add-1:-1])))
            signal_spec = np.fft.fft(signal)
            # signal_spec = signal_spec[]
            if nodes[window]==nodes[-1]:
                if i>=nodes[window]:
                    phases = np.full(num_phases,phase_shifts[i])
                else:
                    phases = np.random.rand(num_phases)*2*np.pi-np.pi
            else:
                if i<nodes[window+1] and i>=nodes[window]:
                    phases = np.full(num_phases,phase_shifts[i])
                else:
                    phases = np.random.rand(num_phases)*2*np.pi-np.pi
            new_phases = np.angle(signal_spec)
            new_phases[positive_frequency_content_indices] = phases
            new_phases[negative_frequency_content_indices] = -phases[::-1]
            tmp = np.abs(signal_spec) * np.exp(1j*new_phases)

            # imaginary part is negligible
            new_signal = np.fft.ifft(tmp).real
            if num_add>0:
                new_signal = new_signal[num_add:-num_add]
            newdata[times[window]:times[window+1],i] = new_signal

    phase_reset_data_all.append(newdata-np.mean(newdata, axis=0))

data_ts = np.concatenate(phase_reset_data_all, axis=0)
data_real_projective_hyperplane = []
data_complex_projective_hyperplane = []
data_grassmann = []
data_spsd = []
data_analytic = []

for scan in range(len(phase_reset_data_all)):
    phases = np.zeros((phase_reset_data_all[scan].shape[0], p))
    magnitudes = np.zeros((phase_reset_data_all[scan].shape[0], p))
    for i in range(p):
        phases[:,i] = np.angle(hilbert(phase_reset_data_all[scan][:,i]))
        magnitudes[:,i] = np.abs(hilbert(phase_reset_data_all[scan][:,i]))
    
    data_analytic.append(magnitudes*np.exp(1j*phases))
    
    U_all_sub = np.zeros((T,p,2))
    L_all_sub = np.zeros((T,2))
    U_all_complex_sub = np.zeros((T,p), dtype=complex)
    for t in range(T):
        c = np.cos(phases[t])
        s = np.sin(phases[t])
        U,S,_ = np.linalg.svd(np.c_[c,s], full_matrices=False)
        # U,S,_ = np.linalg.svd(c+s*1j, full_matrices=False)
        U_all_sub[t] = U
        L_all_sub[t] = S**2
        U_all_complex_sub[t,:] = (c+s*1j)/np.linalg.norm(c+s*1j)
    
    data_real_projective_hyperplane.append(U_all_sub[:,:,0])
    data_complex_projective_hyperplane.append(U_all_complex_sub)
    data_grassmann.append(U_all_sub)
    data_spsd.append(U_all_sub*np.sqrt(L_all_sub[:,None,:]))
    
data_real_projective_hyperplane = np.concatenate(data_real_projective_hyperplane, axis=0)
data_complex_projective_hyperplane = np.concatenate(data_complex_projective_hyperplane, axis=0)
data_grassmann = np.concatenate(data_grassmann, axis=0)
data_spsd = np.concatenate(data_spsd, axis=0)


# use h5py to save this data (U_all)
with h5py.File('paper/data/phase_randomized/narrowband_phase_controlled_116data.h5', 'w') as f:
    f.create_dataset("data_ts", data=data_ts)
    f.create_dataset("data_real_projective_hyperplane", data=data_real_projective_hyperplane)
    f.create_dataset("data_complex_projective_hyperplane", data=data_complex_projective_hyperplane)
    f.create_dataset("data_grassmann", data=data_grassmann)
    f.create_dataset("data_spsd", data=data_spsd)
    f.create_dataset("data_analytic", data=np.concatenate(data_analytic, axis=0))
print('Data saved')