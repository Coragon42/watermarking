import re, csv

input_file = './data/Adaptive/with_emojis.csv'
output_file = './data/Adaptive/without_emojis.csv'

# Function to remove emojis using regex
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["  
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # geometic shapes
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "]", flags=re.UNICODE)
    
    return emoji_pattern.sub(r'', text)

# Open the CSV file
with open(input_file, newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    
    # Prepare the output list to store cleaned rows
    cleaned_rows = []
    
    # Iterate through the rows of the CSV
    for row in reader:
        # Clean each cell in the row
        cleaned_row = [remove_emojis(cell) for cell in row]
        cleaned_rows.append(cleaned_row)

# Write the cleaned data to a new CSV file
with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(cleaned_rows)