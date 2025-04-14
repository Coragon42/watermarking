import csv
import json

input_file = './data/Adaptive/to_redetect.csv'
output_file = input_file[:-3]+'jsonl'

def csv_to_jsonl(csv_file_path, jsonl_file_path):
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile, \
         open(jsonl_file_path, 'w', encoding='utf-8') as jsonlfile:
        
        reader = csv.DictReader(csvfile)
        for row in reader:
            json_line = json.dumps(row)
            jsonlfile.write(json_line + '\n')

# Example usage
csv_to_jsonl(input_file, output_file)