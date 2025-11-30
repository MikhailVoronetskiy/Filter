import numpy as np
import librosa
import soundfile as sf
from scipy.signal import wiener
import matplotlib.pyplot as plt



def compute_snr(clean, test):
    noise = test - clean
    return 10 * np.log10(np.sum(clean ** 2) / np.sum(noise ** 2))


clean_file = "clean_audio.wav"
clean, sr = librosa.load(clean_file, sr=None, mono=True)



noise_power = 0.001
noise = np.sqrt(noise_power) * np.random.randn(len(clean))
noisy = clean + noise

sf.write("noisy_audio.wav", noisy, sr)

snr_before = compute_snr(clean, noisy)
print(f"SNR before filtering: {snr_before:.3f} dB")

window_sizes = range(3, 51, 2)  # odd values from 3 to 19
efficiencies = []

best_snr = -np.inf
best_filtered = None
best_window_size = None

for window_size in window_sizes:
    filtered = wiener(noisy, mysize=window_size)  # no noise parameter: unknown noise scenario
    snr_after = compute_snr(clean, filtered)
    delta_snr = snr_after - snr_before
    efficiency = 100 * delta_snr / abs(snr_before)
    efficiencies.append(efficiency)
    print(f"mysize={window_size}, SNR after={snr_after:.3f} dB, Efficiency={efficiency:.2f}%")

    if snr_after > best_snr:
        best_snr = snr_after
        best_filtered = filtered
        best_window_size = window_size


sf.write("filtered_audio.wav", best_filtered, sr)
plt.figure(figsize=(8, 5))
plt.plot(list(window_sizes), efficiencies, marker='o')
plt.xlabel("Wiener filter window")
plt.ylabel("Efficiency (%)")
plt.title("Wiener Filter Efficiency vs Window size")
plt.grid(True)
plt.tight_layout()
plt.show()
