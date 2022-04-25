import numpy as np
from scipy.signal import hilbert

#Before this programm, you should firstly run the "samples_concat.py" to get the test-datasets!!!
#According to the result of "Kennwerteanalyse", I set 2 creterions to achieve the fault detection.
'''
The creterions are related to the max. frequency-component and the mean of top15 std.-Amplitude. The judge-process includes 2 stages:

1. max_freq: 120-180 -> IR; 86-120 -> NO,IR,BA,OR; others -> unknown
2. (86-120)mean_top15: 0-0.005 -> NO; 0.005-0.01 -> BA; 0.01-0.06 -> IR; 0.1-0.2 -> OR; others -> unknown

'''
# This program is uesd for evaluating the creterion in frequency domain.

#set all parameters
#criterion value
ir_max_freq_low = 120
ir_max_freq_high = 180
others_max_freq_low = 0
others_max_freq_high = 120

no_mean_top15_low = 0
no_mean_top15_high = 0.005
ba_mean_top15_low = 0.005
ba_mean_top15_high = 0.01
ir_mean_top15_low = 0.01
ir_mean_top15_high = 0.06
or_mean_top15_low = 0.1
or_mean_top15_high = 1

#judge_criteria
def judge_value(max_freq,mean_top15):
    if ir_max_freq_low <= max_freq < ir_max_freq_high:
        res = 1
    elif others_max_freq_low < max_freq < others_max_freq_high:
        if no_mean_top15_low< mean_top15 < no_mean_top15_high:
            res = 0
        elif ba_mean_top15_low <= mean_top15 < ba_mean_top15_high:
            res = 2
        elif ir_mean_top15_low <= mean_top15 < ir_mean_top15_high:
            res = 1
        elif or_mean_top15_low < mean_top15 < or_mean_top15_high:
            res = 3
        else:
            res = 4
    else:
        res = 4

    return res

#The test_datasets-shape is [(num_no_samplesx24000),(num_ir_samplesx24000),(num_ba_samplesx24000),(num_or_samplesx24000)]
#0-NO, 1-IR, 2-BA, 3-OR
def envelop_test(filename):
    all_data = np.load('./test_datasets.npy',allow_pickle=True)
    no_data = all_data[0]
    ir_data = all_data[1]
    ba_data = all_data[2]
    or_data = all_data[3]
    sample_size = [len(no_data),len(ir_data),len(ba_data),len(or_data)]

    #The parameters used to evaluate the creterions.
    scores = {0:0,1:0,2:0,3:0}
    right = 0
    mis_right = 0
    unknown = 0

    for typ in range(4):
        for test_data in all_data[typ]:
            # envelope
            analytic_signal = hilbert(test_data)
            amplitude_envelope = np.abs(analytic_signal)

            # FFT of envelope
            xf = np.fft.rfft(amplitude_envelope) / len(test_data)  # fft-transform
            freqs = np.linspace(0, 24000, (len(test_data) // 2 + 1))
            xfp = np.abs(xf)

            freq, amp = freqs[0:250], xfp[0:250]

            #feature-value
            mean_top15 = np.mean(abs(np.sort(-amp))[1:16])
            max_freq = freq[np.argmax(amp[1:])+1]
            res = judge_value(max_freq,mean_top15)
            if res == typ:
                right += 1
                scores[typ] += 1
            elif res == 4:
                unknown += 1
            else:
                mis_right += 1

    results = {'scores':scores,'no_score':scores[0],'ir_score':scores[1],'ba_score':scores[2],'or_score':scores[3],'right':right,'mis_right':mis_right,'unknown':unknown}
    return results,sample_size

if __name__ == "__main__":
    res, sample_size = envelop_test('test_datasets.npy')
    print("The Results--------------------")
    print(res)
    print("sample_size:",sample_size)
    print("positive_rate：",(1-res['unknown']/sum(sample_size)))
    print("judge_acc:",res['right']/(sum(sample_size)-res['unknown']))