import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
            for id, ls in data['tokens_green_red_counts'].items():
                if id not in r_freqs:
                    if ls[1] > 0: # green
                        r_freqs[(id,1)] = [ls[0],ls[1]]
                    else: # red
                        r_freqs[(id,0)] = [ls[0],ls[2]]
                else:
                    if ls[1] > 0: # green
                        r_freqs[(id,1)][1] += ls[1]
                    else: # red
                        r_freqs[(id,0)][1] += ls[2]
    # print(total_tokens)
    for id, ls in r_freqs.items():
        ls[1] /= total_tokens
    return r_freqs

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
        r_freqs_diff[id] = [r_freqs_watermarked[id][0],r_freqs_watermarked[id][1]-r_freqs_unwatermarked.get(id,['',0])[1]]
    else:
        r_freqs_diff[id] = [r_freqs_unwatermarked[id][0],-r_freqs_unwatermarked[id][1]]

r_freqs_diff_desc = sorted(r_freqs_diff.items(), key=lambda x: x[1][1], reverse=True)

top = r_freqs_diff_desc[:100]
bottom = r_freqs_diff_desc[-100:]

# Combine
combined = top + bottom

# Extract data
strings = [f'$_{{_{{{id[0]},{id[1]}}}}}$'+ls[0] for id,ls in combined]
# strings = [ls[0] for id,ls in combined] # runs into double bar issue due to decoding collisions
are_green = [id[1] for id,ls in combined]
diffs = [ls[1] for id,ls in combined]

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
plt.savefig('.\data\Adaptive\compare.png', dpi=300, bbox_inches='tight')
plt.close()
print("Finished!")