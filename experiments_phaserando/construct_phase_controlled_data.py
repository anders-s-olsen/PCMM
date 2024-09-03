import numpy as np
import nibabel as nib
from scipy.signal import butter, filtfilt, hilbert
import h5py
subjects = np.loadtxt('100unrelatedsubjectsIDs.txt', dtype='str')
p = 116
K = 5
T = 1200
num_subs = 1

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="bandpass")
    return b, a

def butter_bandpass_filter(data, lowcut=0.03, highcut=0.07, fs=1 / 0.720, order=5):
# def butter_bandpass_filter(data, lowcut=0.009, highcut=0.08, fs=1 / 0.720, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

atlas = nib.load('data/external/Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1.dlabel.nii')
atlas_data = atlas.get_fdata()
atlas_data = np.round(atlas_data)
atlas_data = atlas_data.astype(int)[0]
filtered_data_all = []

for sub in subjects[:num_subs]:
    img = nib.load('data/raw/'+sub+'/fMRI/rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii')
    data = img.get_fdata()
    data = data - np.mean(data, axis=0)

    GS = np.mean(data[:,:59412], axis=1)
    data = data - GS[:,None]*(data.T@GS).T/(GS.T@GS)
    
            # GS = mean(data(:,1:59412),2);
            # data = data-GS.*(data'*GS)'/(GS'*GS);
    # parcellate data in atlas
    parcellated_data = np.zeros((data.shape[0], atlas_data.max()))
    for i in range(1, atlas_data.max()+1):
        parcellated_data[:,i-1] = np.mean(data[:,atlas_data == i], axis=1)
    # filter data
    filtered_data = np.zeros(parcellated_data.shape)
    for i in range(p):
        filtered_data[:,i] = butter_bandpass_filter(parcellated_data[:,i])
    filtered_data_all.append(filtered_data)

num_repeats = 1 #repeats of clustering pattern state1..state2.. etc

times = np.linspace(0, 1200, K*num_repeats+1, dtype=int)
num_samples = times[1]-times[0]
num_add = num_samples//10
TR = 0.720
f = np.fft.fftfreq(num_samples+num_add*2, 1/TR)
# find indices of frequencies above or equal to 0.008Hz and below or equal to 0.09Hz
positive_frequency_content_indices = np.where((f>=0.009) & (f<=0.08))[0]
negative_frequency_content_indices = np.where((f<=-0.009) & (f>=-0.08))[0]
mid_frequency = np.argmin(np.abs(f-0.05))
num_phases = len(positive_frequency_content_indices)

nodes = [0, 23, 46, 69, 92]
nodes = np.tile(nodes, num_repeats)
specs = []
for window in range(K*num_repeats):
    ref = filtered_data_all[0][times[window]:times[window+1],nodes[window]]
    if num_add>0:
        ref = np.concatenate((2*ref[0]-np.flip(ref[1:num_add+1]),ref,2*ref[-1]-np.flip(ref[-num_add-1:-1])))
    specs.append(np.fft.fft(ref))

phase_reset_data_all = []
phase_shifts = (np.random.rand(p)*2*np.pi-np.pi)*1 #within-cluster phase shifts equal across subjects
for sub in range(num_subs):
    newdata = np.zeros((T,p))
    for window in range(K*num_repeats):
        for i in range(p):
            signal = filtered_data_all[sub][times[window]:times[window+1],i]
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
            # tmp = np.abs(signal_spec) * np.exp(1j*new_phases)
            tmp = np.abs(specs[window]) * np.exp(1j*new_phases)
            # new_amps = np.zeros(num_samples+num_add*2)
            # new_amps[mid_frequency] = 1
            # tmp = new_amps * np.exp(1j*new_phases)
            
            # imaginary part is negligible
            new_signal = np.fft.ifft(tmp).real
            if num_add>0:
                new_signal = new_signal[num_add:-num_add]
            newdata[times[window]:times[window+1],i] = new_signal

    phase_reset_data_all.append(newdata)

U_all = []
L_all = []
U_all_complex = []
L_all_complex = []
for sub in range(num_subs):
    phases = np.zeros((phase_reset_data_all[sub].shape[0], p))
    for i in range(p):
        phases[:,i] = np.angle(hilbert(phase_reset_data_all[sub][:,i]))
    U_all_sub = np.zeros((T,p,2))
    L_all_sub = np.zeros((T,2))
    U_all_complex_sub = np.zeros((T,p,1), dtype=complex)
    L_all_complex_sub = np.zeros((T,1))
    for t in range(T):
        c = np.cos(phases[t])
        s = np.sin(phases[t])
        U,S,_ = np.linalg.svd(np.c_[c,s], full_matrices=False)
        # U,S,_ = np.linalg.svd(c+s*1j, full_matrices=False)
        U_all_sub[t] = U
        L_all_sub[t] = S**2
        U_all_complex_sub[t,:,0] = (c+s*1j)/np.linalg.norm(c+s*1j)
        L_all_complex_sub[t,0] = p
    U_all.append(U_all_sub)
    L_all.append(L_all_sub)
    U_all_complex.append(U_all_complex_sub)
    L_all_complex.append(L_all_complex_sub)
U_tmp = np.concatenate(U_all, axis=0)
L_tmp = np.concatenate(L_all, axis=0)
U_complex_tmp = np.concatenate(U_all_complex, axis=0)
L_complex_tmp = np.concatenate(L_all_complex, axis=0)

# use h5py to save this data (U_all)
with h5py.File('data/synthetic/phase_narrowband_controlled_116data_eida.h5', 'w') as f:
# with h5py.File('data/synthetic/phase_amplitude_controlled_116data_eida.h5', 'w') as f:
# with h5py.File('data/synthetic/phase_controlled_116data_eida.h5', 'w') as f:
    f.create_dataset("U", data=U_tmp)
    f.create_dataset("L", data=L_tmp)
print('Data saved')

with h5py.File('data/synthetic/complex_phase_narrowband_controlled_116data_eida.h5', 'w') as f:
# with h5py.File('data/synthetic/complex_phase_amplitude_controlled_116data_eida.h5', 'w') as f:
# with h5py.File('data/synthetic/complex_phase_controlled_116data_eida.h5', 'w') as f:
    f.create_dataset("U", data=U_complex_tmp)
    f.create_dataset("L", data=L_complex_tmp)