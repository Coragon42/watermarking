import json
import csv

input_file = './data/Adaptive/prompts_v0,facebook-opt-1.3b,0,0.5,300,None,None,0.9,6.0,0,200,v0.jsonl'
# input_file = './data/Adaptive/to_redetect_unwatermarked.jsonl'
output_file = input_file[:-5]+'csv'

# Read JSONL and collect records
records = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            dct = json.loads(line)
            flat = {}
            flat["time_completed"] = dct["time_completed"]
            flat["prefix"] = dct["prefix"]
            flat["gold_completion"] = dct["gold_completion"]
            flat["gen_completion"] = dct["gen_completion"][0]
            flat["too_short"] = dct["too_short"]
            flat["z-score_original"] = dct["z-score"][0]
            flat["z-score_unique"] = dct["z-score"][1]
            flat["p-value_original"] = dct["p-value"][0]
            flat["p-value_unique"] = dct["p-value"][1]
            flat["wm_pred_original"] = dct["wm_pred"][0]
            flat["wm_pred_unique"] = dct["wm_pred"][1]
            flat["gen_length_original"] = dct["gen_length"][0]
            flat["gen_length_unique"] = dct["gen_length"][1]
            flat["num_green_original"] = dct["num_green"][0]
            flat["num_green_unique"] = dct["num_green"][1]
            flat["are_tokens_green"] = dct["are_tokens_green"]
            records.append(flat)

# Write to CSV
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)