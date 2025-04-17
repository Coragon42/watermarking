import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

watermarked = '.\data\Adaptive\Unigram,Adaptive,facebook-opt-1.3b,2.0,0.5,300,None,None,0.9,6.0,0,200,v0.jsonl'
unwatermarked = '.\data\Adaptive\Unigram,Adaptive,facebook-opt-1.3b,0,0.5,300,None,None,0.9,6.0,0,200,v0.jsonl'

def aggregate(file, start, stop):
    r_freqs: dict[int,[str,int,float]] = {} # {id:[string,is_green,r_freq]}
    total_tokens = 0
    with open(file) as f:
        for row, line in enumerate(f): # zero-indexed, unlike viewing jsonl
            if row < start or row >= stop:
                continue
            data = json.loads(line)
            for id, tup in data['are_tokens_green'].items():
                if id not in r_freqs:
                    r_freqs[id] = tup
                else:
                    r_freqs[id][2] += tup[2]
                total_tokens += tup[2]
    # print(total_tokens)
    for id, tup in r_freqs.items():
        tup[2] /= total_tokens
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
        r_freqs_diff[id] = [r_freqs_watermarked[id][0],r_freqs_watermarked[id][1],r_freqs_watermarked[id][2]-r_freqs_unwatermarked.get(id,['',0,0])[2]]
    else:
        r_freqs_diff[id] = [r_freqs_unwatermarked[id][0],r_freqs_unwatermarked[id][1],-r_freqs_unwatermarked[id][2]]

r_freqs_diff_desc = sorted(r_freqs_diff.items(), key=lambda x: x[1][2], reverse=True)

top = r_freqs_diff_desc[:100]
bottom = r_freqs_diff_desc[-100:]

# Combine
combined = top + bottom

# Extract data
ids = [f'$_{{_{{{id}}}}}$'+tup[0] for id,tup in combined]
# strings = [tup[0] for id,tup in combined] # runs into double bar issue due to decoding collisions
are_green = [tup[1] for id,tup in combined]
diffs = [tup[2] for id,tup in combined]

colors = ['green' if b==1 else 'red' for b in are_green]

plt.figure(figsize=(30, 6))
bars = plt.bar(ids, diffs, color=colors)
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