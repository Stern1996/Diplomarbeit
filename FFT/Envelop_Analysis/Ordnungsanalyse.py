import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from scipy.signal import hilbert
import random
from scipy.fft import fft, fftfreq, fftshift
from scipy.stats import skew


def envelop(filename, time, rpm):
    # readin data
    data = scio.loadmat(filename)
    num = filename.strip(".mat")
    if int(num) < 100:
        index = "X" + "0" + num + "_DE_time"
    else:
        index = "X" + num + "_DE_time"
    data = data[index]
    data = data.flatten()

    # get the data
    samplingrate = 48000
    dt = 1 / samplingrate
    start = random.randint(0, (data.shape[0] - int(samplingrate * time) - 1))
    data = data[start:(start + int(samplingrate * time))]
    t = np.linspace(0, (samplingrate * time - 1) * dt, int(samplingrate * time))

    # envelope
    analytic_signal = hilbert(data)
    amplitude_envelope = np.abs(analytic_signal)

    # FFT of envelope
    fft_size = int(samplingrate * time)
    xf = np.fft.rfft(amplitude_envelope) / fft_size  # fft-transform
    freqs = np.linspace(0, 24000, (fft_size // 2 + 1)) * 60 / rpm
    xfp = np.abs(xf)

    fig, (ax1) = plt.subplots(nrows=1)
    # ax0.plot(t, data, label='signal')
    # ax0.plot(t, amplitude_envelope, label='envelope')
    # ax0.set_xlabel("time in seconds")
    # ax0.legend()

    # only draw the top 250 frequency component, frequency interval [0,(freqs[1]-freqs[0])*250]
    ax1.plot(freqs[0:250], xfp[0:250])
    ax1.set_xlabel("Ord. in x n")
    ax1.set_ylabel("mm/s")
    plt.savefig(num + "_Hüllspektrum.jpg")
    plt.show()

    return freqs[0:250], xfp[0:250]

if __name__ == "__main__":
    freq, amp = envelop("241.mat", 0.5, 1730)