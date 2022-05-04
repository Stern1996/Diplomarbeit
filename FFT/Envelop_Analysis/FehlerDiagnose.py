import numpy as np
from scipy.signal import hilbert
import json

#Before this programm, you should firstly run the "samples_concat.py" to get the test-datasets!!!
#According to the result of "Kennwerteanalyse", I set 2 creterions to achieve the fault detection.
'''
The creterions are related to the max. frequency-component and the mean of top15 std.-Amplitude. The judge-process includes 2 stages:

1. max_freq: 120-180 -> IR; 0-120 -> NO,IR,BA,OR; others -> abnormal
2. (0-120)mean_top15: 0-0.005 -> NO; 0.005-0.01 -> BA; 0.01-0.06 -> IR; 0.1-0.2 -> OR; others -> abnormal

'''
# This program is uesd for evaluating the creterion in frequency domain.

# read index data
with open('data/index.json') as json_file:
    file_json = json.load(json_file)

#show the criterion information
def show_criterions():
    with open('data/criterion.json') as json_file:
        criterion_json = json.load(json_file)
    criterion_names = list(criterion_json.keys())
    for name in criterion_names:
        print("criterion plan:",name)
        print(criterion_json[name])
        print("=======================")
    return criterion_names

# read criterion data
def read_criterion(criterion_name):
    with open('data/criterion.json') as json_file:
        criterion_json = json.load(json_file)
    return criterion_json[criterion_name]

#add new criterion in json, max_freq => ir?, mean-top15 => no?ba?ir?or?
def add_criterion(name):
    criterion_name = name
    print("========Please set the parameters=======")
    ir_max_freq_low = float(input("ir_max_freq_low:"))
    ir_max_freq_high = float(input("ir_max_freq_high:"))
    others_max_freq_low = float(input("others_max_freq_low:"))
    others_max_freq_high = float(input("others_max_freq_high:"))

    no_mean_top15_low = float(input("no_mean_top15_low:"))
    no_mean_top15_high = float(input("no_mean_top15_high:"))
    ba_mean_top15_low = float(input("ba_mean_top15_low:"))
    ba_mean_top15_high = float(input("ba_mean_top15_high:"))
    ir_mean_top15_low = float(input("ir_mean_top15_low:"))
    ir_mean_top15_high = float(input("ir_mean_top15_high:"))
    or_mean_top15_low = float(input("or_mean_top15_low:"))
    or_mean_top15_high = float(input("or_mean_top15_high:"))

    new_criterion = {criterion_name:{"ir_max_freq_low":ir_max_freq_low,"ir_max_freq_high":ir_max_freq_high,"others_max_freq_low":others_max_freq_low,"others_max_freq_high":others_max_freq_high,
                                     "no_mean_top15_low":no_mean_top15_low,"no_mean_top15_high":no_mean_top15_high,
                                     "ba_mean_top15_low":ba_mean_top15_low,"ba_mean_top15_high":ba_mean_top15_high,
                                     "ir_mean_top15_low":ir_mean_top15_low,"ir_mean_top15_high":ir_mean_top15_high,
                                     "or_mean_top15_low":or_mean_top15_low,"or_mean_top15_high":or_mean_top15_high}}

    with open('data/criterion.json') as json_file:
        criterion_json = json.load(json_file)
    criterion_json.update(new_criterion)
    with open('data/criterion.json', "w") as json_file:
        json.dump(criterion_json, json_file)


#judge_criteria，return int-result,input-2criterions
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
def envelop_test(criterion):
    all_data = np.load('./data/database.npy',allow_pickle=True)
    all_data = all_data.item()
    file_names = list(all_data.keys())

    results = {}

    for file in file_names:
        typ = list(all_data[file].keys())[0]
        data = list(all_data[file].values())[0]
        num_samples = len(data)
        # processing each sample-data
        right = 0

        for sample_data in data:
            # envelope
            analytic_signal = hilbert(sample_data)
            amplitude_envelope = np.abs(analytic_signal)

            # FFT of envelope
            xf = np.fft.rfft(amplitude_envelope) / len(sample_data)  # fft-transform
            freqs = np.linspace(0, 24000, (len(sample_data) // 2 + 1))
            xfp = np.abs(xf)

            freq, amp = freqs[0:250], xfp[0:250]

            # feature-value
            mean_top15 = np.mean(abs(np.sort(-amp))[1:16])
            max_freq = freq[np.argmax(amp[1:]) + 1]
            res = judge_value(max_freq, mean_top15)

            if res == typ:
                right += 1

        acc = right / num_samples
        results.update({file: acc})

    results = {criterion: results}
    with open("data/results.json", "w+") as f:
        json.dump(results, f)

if __name__ == "__main__":
    # set the criterion
    criterion_names = show_criterions()
    criterion_plan = input("input the name of criterion plan:\n")
    if criterion_plan in criterion_names:
        criterions = read_criterion(criterion_plan)
    else:
        print("There are no settled plans to use, create new criterion-plan......")
        add_criterion(criterion_plan)
        criterions = read_criterion(criterion_plan)

    # set all parameters
    # criterion value
    ir_max_freq_low = criterions["ir_max_freq_low"]
    ir_max_freq_high = criterions["ir_max_freq_high"]
    others_max_freq_low = criterions["others_max_freq_low"]
    others_max_freq_high = criterions["others_max_freq_high"]

    no_mean_top15_low = criterions["no_mean_top15_low"]
    no_mean_top15_high = criterions["no_mean_top15_high"]
    ba_mean_top15_low = criterions["ba_mean_top15_low"]
    ba_mean_top15_high = criterions["ba_mean_top15_high"]
    ir_mean_top15_low = criterions["ir_mean_top15_low"]
    ir_mean_top15_high = criterions["ir_mean_top15_high"]
    or_mean_top15_low = criterions["or_mean_top15_low"]
    or_mean_top15_high = criterions["or_mean_top15_high"]

    print("The criterion is:", criterions)
    print("==========================")
    envelop_test(criterion_plan)
    print("The diagnosis-accuracy in each dataset:")
    with open("data/results.json") as f:
        results = json.load(f)
        results = results[criterion_plan]
        for file in results:
            print("accuracy in "+file,results[file])
