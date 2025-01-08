import csv
#

input_file = 'Data/gold_BUG.csv'
output_file = 'gold_BUG_balanced_100.txt'
detailed_output_file ='gold_BUG_balanced_100_tagged.txt'

max_per_group = 25

conditions = {
    ('Male', 1): [],
    ('Female', 1): [],
    ('Male', -1): [],
    ('Female', -1): []
}


bad_words = ["person","child","writer"]
max_per_group = 25

# Read the CSV and filter rows
with open(input_file, 'r', encoding='utf-8') as csv_file:
    reader = csv.DictReader(csv_file)  # Read as a dictionary to access columns by name
    for row in reader:
        # Only consider rows from Wikipedia
        if 'wikipedia' in row['corpus']:
            # Get the relevant fields
            predicted_gender = row['predicted gender']
            stereotype = int(row['stereotype'])  # Convert stereotype to an integer

            # Check if the row matches any condition and hasn't reached the limit
            key = (predicted_gender, stereotype)
            if key in conditions and len(conditions[key]) < max_per_group and (bad_words[0] not in row['sentence_text'] and bad_words[1] not in row['sentence_text'] and bad_words[2] not in row['sentence_text']):
                conditions[key].append((row['sentence_text'], predicted_gender, stereotype))


with open(output_file, 'w', encoding='utf-8') as txt_file:
    for group, rows in conditions.items():
        for sentence, _, _ in rows:
            txt_file.write(sentence + '\n')

with open(detailed_output_file, 'w', encoding='utf-8') as detailed_txt_file:
    for group, rows in conditions.items():
        for sentence, gender, stereotype in rows:
            detailed_txt_file.write(f"{sentence}\t{gender}\t{stereotype}\n")
