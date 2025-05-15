import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
import random
from time import time
import csv

watermarked = r'.\data\Adaptive\extended,Adaptive,facebook-opt-1.3b,2.0,0.5,300,4.0,25,simple_1,,v0.jsonl'
unwatermarked = r'.\data\Adaptive\extended,Adaptive,facebook-opt-1.3b,0,0.5,300,4.0,25,simple_1,,v0.jsonl'

def aggregate(file, start, stop):
    # need to distinguish between green/red for same id because each token can be both green/red on different occasions within same output
    r_freqs: dict[(int,int),[str,float]] = {} # {(id,is_green):[string,r_freq]}
    total_tokens = 0
    with open(file,"r") as f:
        for row, line in enumerate(f): # zero-indexed, unlike viewing jsonl
            if row < start or row >= stop:
                continue
            data = json.loads(line)
            total_tokens += data['num_tokens_scored'][0]
            for id, tup in data['tokens_green_red_counts'].items():
                if id not in r_freqs:
                    if tup[1] > 0: # green
                        r_freqs[(id,1)] = [tup[0],tup[1]]
                    else: # red
                        r_freqs[(id,0)] = [tup[0],tup[2]]
                else:
                    if tup[1] > 0: # green
                        r_freqs[(id,1)][1] += tup[1]
                    else: # red
                        r_freqs[(id,0)][1] += tup[2]
    # print(total_tokens)
    for id, tup in r_freqs.items():
        tup[1] /= total_tokens
    return r_freqs

def aggregate_random(file, start, stop, sample):
    indices = set(random.sample(range(start,stop),sample))
    # need to distinguish between green/red for same id because each token can be both green/red on different occasions within same output
    r_freqs: dict[(int,int),[str,float]] = {} # {(id,is_green):[string,r_freq]}
    total_tokens = 0
    with open(file,"r") as f:
        for row, line in enumerate(f): # zero-indexed, unlike viewing jsonl
            if row not in indices:
                continue
            data = json.loads(line)
            total_tokens += data['num_tokens_scored'][0]
            for id, tup in data['tokens_green_red_counts'].items():
                if id not in r_freqs:
                    if tup[1] > 0: # green
                        r_freqs[(id,1)] = [tup[0],tup[1]]
                    else: # red
                        r_freqs[(id,0)] = [tup[0],tup[2]]
                else:
                    if tup[1] > 0: # green
                        r_freqs[(id,1)][1] += tup[1]
                    else: # red
                        r_freqs[(id,0)][1] += tup[2]
    # print(total_tokens)
    for id, tup in r_freqs.items():
        tup[1] /= total_tokens
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
            r_freqs_diff[id] = [r_freqs_watermarked[id][0],r_freqs_watermarked[id][1]-r_freqs_unwatermarked.get(id,['',0])[1]]
        else:
            r_freqs_diff[id] = [r_freqs_unwatermarked[id][0],-r_freqs_unwatermarked[id][1]]

    r_freqs_diff_desc = sorted(r_freqs_diff.items(), key=lambda x: x[1][1], reverse=True)

    top = r_freqs_diff_desc[:cut]
    bottom = r_freqs_diff_desc[-cut:]

    top_prop = 0
    for tup in top:
        top_prop += tup[0][1] # is_green
    top_prop /= cut
    bottom_prop = 0
    for tup in bottom:
        bottom_prop += tup[0][1] # is_green
    bottom_prop /= cut
    return top_prop,bottom_prop

def grid_search():
    start_list = [0,500] # LQFA, OpenGen
    sample_list = [5,10,20] + list(range(50,501,50))
    cut_list = [5,10,20] + list(range(50,1001,50))
    max = 0
    max_params = []
    with open(".\data\Adaptive\compare_soft_" + str(int(time())) + ".csv","w",newline="") as f:
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
                    # 598 rows
                    print(start,stop,sample,cut,avg_top_prop,avg_bottom_prop,avg_accuracy)
                    writer.writerow([start,stop,sample,cut,avg_top_prop,avg_bottom_prop,avg_accuracy])
                    f.flush()
                    if avg_accuracy > max:
                        max = avg_accuracy
                        max_params = [start,stop,sample,cut]
    return max,max_params

def main():
    r_freqs_watermarked = aggregate(watermarked,500,1000)
    r_freqs_unwatermarked = aggregate(unwatermarked,500,1000)

    all_ids = set(r_freqs_watermarked.keys()) | set (r_freqs_unwatermarked.keys())
    r_freqs_diff = {}
    for id in all_ids:
        if id in r_freqs_watermarked:
            r_freqs_diff[id] = [r_freqs_watermarked[id][0],r_freqs_watermarked[id][1]-r_freqs_unwatermarked.get(id,['',0])[1]]
        else:
            r_freqs_diff[id] = [r_freqs_unwatermarked[id][0],-r_freqs_unwatermarked[id][1]]

    r_freqs_diff_desc = sorted(r_freqs_diff.items(), key=lambda x: x[1][1], reverse=True)

    cut=15
    fontsize=16
    top = r_freqs_diff_desc[:cut]
    bottom = r_freqs_diff_desc[-cut:]

    # Combine
    combined = top + bottom

    # Extract data
    # strings = [f'$_{{_{{{id[0]},{id[1]}}}}}$'+tup[0] for id,tup in combined]
    strings = [item[1][0]+f'$_{{_{{{i}}}}}$' for i,item in enumerate(combined)]
    # strings = [tup[0] for id,tup in combined] # runs into double bar issue due to decoding collisions
    are_green = [id[1] for id,tup in combined]
    diffs = [tup[1] for id,tup in combined]

    colors = ['green' if b==1 else 'red' for b in are_green]

    # vertical, cut=15, fontsize=16
    # plt.figure(figsize=(4, 15),dpi=300)
    # frame=plt.gca()
    # frame.grid(which='major', axis='x', color='gray', linestyle='--', linewidth=0.5, zorder=0)
    # frame.grid(which='minor', axis='x', color='lightgray', linestyle='--', linewidth=0.5, zorder=0)
    # frame.minorticks_on()
    # frame.set_ymargin(0.01)
    # bars = plt.barh(strings, diffs, color=colors,zorder=3)
    # plt.axvline(0, color='black', linewidth=0.8)

    # legend_patches = [
    #     Patch(color='green', label='green list'),
    #     Patch(color='red', label='red list')
    # ]
    # plt.legend(loc='upper left',handles=legend_patches,fontsize=fontsize,bbox_to_anchor=(1, 1))

    # # Aesthetics
    # plt.xticks(fontsize=fontsize)
    # plt.yticks(fontsize=fontsize)
    # plt.xlabel("rel. freq. diff. (wm.-unwm.)",fontsize=fontsize)
    # # plt.xscale('symlog', linthresh=1e-4)
    # major_ticks = [-4e-4,-2e-4,0,2e-4,4e-4]
    # minor_ticks = [-3e-4,-1e-4,1e-4,3e-4]
    # custom_labels = ['-4e-4','-2e-4','0','2e-4','4e-4']
    # frame.xaxis.set_major_locator(ticker.FixedLocator(major_ticks))
    # frame.xaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
    # frame.set_xticklabels(custom_labels)
    # frame.invert_yaxis()
    # plt.title("Largest Relative Frequency Differences by Token (soft)",fontsize=fontsize)
    # plt.tight_layout(rect=[0, 0, 0.85, 1])


    # vertical, cut=15, fontsize=16
    plt.figure(figsize=(5, 10),dpi=300)
    frame = plt.gca()
    frame.grid(which='major', axis='x', color='gray', linestyle='--', linewidth=0.5, zorder=0)
    frame.grid(which='minor', axis='x', color='lightgray', linestyle='--', linewidth=0.5, zorder=0)
    frame.minorticks_on()
    frame.set_ymargin(0.01)
    bars = plt.barh(strings, diffs, color=colors,zorder=3)
    plt.axvline(0, color='black', linewidth=0.8)

    legend_patches = [
        Patch(color='green', label='green list'),
        Patch(color='red', label='red list')
    ]
    plt.legend(loc='upper left',handles=legend_patches,fontsize=fontsize,bbox_to_anchor=(1, 1))

    # Aesthetics
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("rel. freq. diff. (wm.-unwm.)",fontsize=fontsize)
    # plt.xscale('symlog', linthresh=1e-4)
    major_ticks = [-4e-4,-2e-4,0,2e-4,4e-4]
    minor_ticks = [-3e-4,-1e-4,1e-4,3e-4]
    custom_labels = ['-4e-4','-2e-4','0','2e-4','4e-4']
    frame.xaxis.set_major_locator(ticker.FixedLocator(major_ticks))
    frame.xaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
    frame.set_xticklabels(custom_labels)
    frame.invert_yaxis()
    plt.title("Largest Relative Frequency Differences by Token (soft)",fontsize=fontsize)
    plt.tight_layout(rect=[0, 0, 1.7, 1])

    frame.axes.get_yaxis().set_visible(False)
    for i,rect in enumerate(frame.patches):
        if i<cut:
            frame.text(-0.00002,rect.get_y()+0.65,strings[i],ha='right',fontsize=fontsize)
        else:
            frame.text(0.00002,rect.get_y()+0.65,strings[i],ha='left',fontsize=fontsize)


    # horizontal, cut=50, fontsize=16
    # plt.figure(figsize=(35, 5),dpi=300)
    # frame=plt.gca()
    # frame.grid(which='major', axis='y', color='gray', linestyle='--', linewidth=0.5, zorder=0)
    # frame.grid(which='minor', axis='y', color='lightgray', linestyle='--', linewidth=0.5, zorder=0)
    # frame.set_xmargin(0.01)
    # frame.minorticks_on()
    # bars = plt.bar(strings, diffs, color=colors,zorder=3)
    # plt.axhline(0, color='black', linewidth=0.8)

    # legend_patches = [
    #     Patch(color='green', label='green list'),
    #     Patch(color='red', label='red list')
    # ]
    # plt.legend(loc='upper left',handles=legend_patches,fontsize=fontsize,bbox_to_anchor=(1, 1))

    # # Aesthetics
    # plt.xticks(rotation=45,ha='right',rotation_mode='anchor',fontsize=fontsize)
    # plt.yticks(fontsize=fontsize)
    # plt.ylabel("rel. freq. diff. (wm.-unwm.)",fontsize=fontsize)
    # # plt.xscale('symlog', linthresh=1e-4)
    # major_ticks = [-4e-4,-2e-4,0,2e-4,4e-4]
    # minor_ticks = [-3e-4,-1e-4,1e-4,3e-4]
    # custom_labels = ['-4e-4','-2e-4','0','2e-4','4e-4']
    # frame.yaxis.set_major_locator(ticker.FixedLocator(major_ticks))
    # frame.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
    # frame.set_yticklabels(custom_labels)
    # # frame.invert_yaxis()
    # plt.title("Largest Relative Frequency Differences by Token (soft)",fontsize=fontsize)
    # plt.tight_layout(rect=[0, 0, 0.8, 1])


    # horizontal, cut=50, fontsize=16
    # plt.figure(figsize=(35, 6.5),dpi=300)
    # frame = plt.gca()
    # frame.grid(which='major', axis='y', color='gray', linestyle='--', linewidth=0.5, zorder=0)
    # frame.grid(which='minor', axis='y', color='lightgray', linestyle='--', linewidth=0.5, zorder=0)
    # frame.set_xmargin(0.01)
    # frame.minorticks_on()
    # bars = plt.bar(strings, diffs, color=colors,zorder=3)
    # plt.axhline(0, color='black', linewidth=0.8)

    # legend_patches = [
    #     Patch(color='green', label='green list'),
    #     Patch(color='red', label='red list')
    # ]
    # plt.legend(loc='upper left',handles=legend_patches,fontsize=fontsize,bbox_to_anchor=(1, 1))

    # # Aesthetics
    # plt.xticks(rotation=45,ha='right',rotation_mode='anchor',fontsize=fontsize)
    # plt.yticks(fontsize=fontsize)
    # plt.ylabel("rel. freq. diff. (wm.-unwm.)",fontsize=fontsize)
    # # # plt.xscale('symlog', linthresh=1e-4)
    # major_ticks = [-4e-4,-2e-4,0,2e-4,4e-4]
    # minor_ticks = [-3e-4,-1e-4,1e-4,3e-4]
    # custom_labels = ['-4e-4','-2e-4','0','2e-4','4e-4']
    # frame.yaxis.set_major_locator(ticker.FixedLocator(major_ticks))
    # frame.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
    # frame.set_yticklabels(custom_labels)
    # # frame.invert_yaxis()
    # plt.title("Largest Relative Frequency Differences by Token (soft)",fontsize=fontsize)
    # plt.tight_layout(rect=[0, 0, 0.8, 1])

    # frame.axes.get_xaxis().set_visible(False)
    # for i,rect in enumerate(frame.patches):
    #     if i<cut:
    #         frame.text(rect.get_x()+0.65,-0.00002,strings[i],ha='right',fontsize=fontsize,rotation=90,rotation_mode='anchor')
    #     else:
    #         frame.text(rect.get_x()+0.65,0.00002,strings[i],ha='left',fontsize=fontsize,rotation=90,rotation_mode='anchor')

    plt.savefig('.\data\Adaptive\compare_soft.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Finished!")

if __name__ == '__main__':
    # print(grid_search())
    main()