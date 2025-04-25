import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import random
from time import time
import csv

watermarked = r'.\data\Adaptive\Unigram,Adaptive,facebook-opt-1.3b,2.0,0.5,300,None,None,0.9,6.0,0,200,v0.jsonl'
unwatermarked = r'.\data\Adaptive\Unigram,Adaptive,facebook-opt-1.3b,0,0.5,300,None,None,0.9,6.0,0,200,v0.jsonl'

def aggregate(file, start, stop):
    r_freqs: dict[int,[str,int,float]] = {} # {id:[string,is_green,r_freq]}
    total_tokens = 0
    with open(file,"r") as f:
        for row, line in enumerate(f): # zero-indexed, unlike viewing jsonl
            if row < start or row >= stop:
                continue
            data = json.loads(line)
            total_tokens += data['gen_length'][0]
            for id, tup in data['are_tokens_green'].items():
                if id not in r_freqs:
                    r_freqs[id] = tup
                else:
                    r_freqs[id][2] += tup[2]
    # print(total_tokens)
    for id, tup in r_freqs.items():
        tup[2] /= total_tokens
    return r_freqs

def aggregate_random(file, start, stop, sample):
    indices = set(random.sample(range(start,stop),sample))
    r_freqs: dict[int,[str,int,float]] = {} # {id:[string,is_green,r_freq]}
    total_tokens = 0
    with open(file,"r") as f:
        for row, line in enumerate(f): # zero-indexed, unlike viewing jsonl
            if row not in indices:
                continue
            data = json.loads(line)
            total_tokens += data['gen_length'][0]
            for id, tup in data['are_tokens_green'].items():
                if id not in r_freqs:
                    r_freqs[id] = tup
                else:
                    r_freqs[id][2] += tup[2]
    # print(total_tokens)
    for id, tup in r_freqs.items():
        tup[2] /= total_tokens
    return r_freqs

def test(start,stop,sample,cut):
    # start: start index (starting from 0, unlike when viewing jsonl manually) of output rows
    # stop: stop index exclusive
    # sample: sample size of outputs to randomly select
    # cut: take this many tokens from top/bottom of sorted differences
    # returns: pair of percentages of green among positive differences vs. among negative differences (want 1 and 0 respectively)
    r_freqs_watermarked = aggregate_random(watermarked,start,stop,sample)
    r_freqs_unwatermarked = aggregate_random(unwatermarked,start,stop,sample)

    all_ids = set(r_freqs_watermarked.keys()) | set (r_freqs_unwatermarked.keys())
    r_freqs_diff = {}
    for id in all_ids:
        if id in r_freqs_watermarked:
            r_freqs_diff[id] = [r_freqs_watermarked[id][0],r_freqs_watermarked[id][1],r_freqs_watermarked[id][2]-r_freqs_unwatermarked.get(id,['',0,0])[2]]
        else:
            r_freqs_diff[id] = [r_freqs_unwatermarked[id][0],r_freqs_unwatermarked[id][1],-r_freqs_unwatermarked[id][2]]

    r_freqs_diff_desc = sorted(r_freqs_diff.items(), key=lambda x: x[1][2], reverse=True)

    top = r_freqs_diff_desc[:cut]
    bottom = r_freqs_diff_desc[-cut:]

    top_prop = 0
    for tup in top:
        top_prop += tup[1][1] # is_green
    top_prop /= cut
    bottom_prop = 0
    for tup in bottom:
        bottom_prop += tup[1][1] # is_green
    bottom_prop /= cut
    return top_prop,bottom_prop

def grid_search():
    start_list = [0,500,1000,1500] # LQFA, OpenGen, fixed1, fixed2
    sample_list = [5,10,20] + list(range(50,501,50))
    cut_list = [5,10,20] + list(range(50,1001,50))
    max = 0
    max_params = []
    with open(".\data\Adaptive\compare_Unigram_" + str(int(time())) + ".csv","w",newline="") as f:
        writer = csv.writer(f)
        for start in start_list:
            stop = start + 500
            for sample in sample_list:
                for cut in cut_list:
                    avg_top_prop = 0
                    avg_bottom_prop = 0
                    avg_accuracy = 0
                    repeats = 50
                    for i in range(repeats):
                        props = test(start,stop,sample,cut)
                        avg_top_prop += props[0]
                        avg_bottom_prop += props[1]
                        avg_accuracy += (props[0]+(1-props[1]))/2
                    avg_top_prop /= repeats
                    avg_bottom_prop /= repeats
                    avg_accuracy /= repeats
                    # 1196 rows
                    print(start,stop,sample,cut,avg_top_prop,avg_bottom_prop,avg_accuracy)
                    writer.writerow([start,stop,sample,cut,avg_top_prop,avg_bottom_prop,avg_accuracy])
                    f.flush()
                    if avg_accuracy > max:
                        max = avg_accuracy
                        max_params = [start,stop,sample,cut]
    return max,max_params

def main():
    r_freqs_watermarked = aggregate(watermarked,0,20)
    r_freqs_unwatermarked = aggregate(unwatermarked,0,20)

    # r_freqs_watermarked = aggregate(watermarked,0,50)
    # r_freqs_unwatermarked = aggregate(unwatermarked,0,50)

    # r_freqs_watermarked = aggregate(watermarked,1001,1051)
    # r_freqs_unwatermarked = aggregate(unwatermarked,1001,1051)

    all_ids = set(r_freqs_watermarked.keys()) | set (r_freqs_unwatermarked.keys())
    r_freqs_diff = {}
    for id in all_ids:
        if id in r_freqs_watermarked:
            r_freqs_diff[id] = [r_freqs_watermarked[id][0],r_freqs_watermarked[id][1],r_freqs_watermarked[id][2]-r_freqs_unwatermarked.get(id,['',0,0])[2]]
        else:
            r_freqs_diff[id] = [r_freqs_unwatermarked[id][0],r_freqs_unwatermarked[id][1],-r_freqs_unwatermarked[id][2]]

    r_freqs_diff_desc = sorted(r_freqs_diff.items(), key=lambda x: x[1][2], reverse=True) # list of tuples

    top = r_freqs_diff_desc[:100]
    bottom = r_freqs_diff_desc[-100:]

    # Combine
    combined = top + bottom

    # Extract data
    strings = [f'$_{{_{{{id}}}}}$'+tup[0] for id,tup in combined]
    # strings = [tup[0] for id,tup in combined] # runs into double bar issue due to decoding collisions
    are_green = [tup[1] for id,tup in combined]
    diffs = [tup[2] for id,tup in combined]

    colors = ['green' if b==1 else 'red' for b in are_green]

    plt.figure(figsize=(30, 6))
    bars = plt.bar(strings, diffs, color=colors)
    plt.axhline(0, color='black', linewidth=0.8)

    legend_patches = [
        Patch(color='green', label='green list'),
        Patch(color='red', label='red list')
    ]
    plt.legend(handles=legend_patches)

    # Aesthetics
    plt.xticks(rotation=90, ha='right')
    plt.ylabel("Relative Frequency Difference (watermarked-unwatermarked)")
    plt.yscale('symlog', linthresh=1e-3)
    plt.title("Largest Relative Frequency Differences (watermarked-unwatermarked)")
    plt.tight_layout()
    plt.savefig('.\data\Adaptive\compare_Unigram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Finished!")

if __name__ == '__main__':
    print(grid_search())
    # main()