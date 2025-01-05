import csv
#
# # Input and output file paths
# input_file = 'gold_BUG.csv'
# output_file = 'gold_BUG_other_10.txt'
#
# # Specify the row range
# start_row = 11
# end_row = 21
#
# # Read the CSV and write the specified range of rows
# with open(input_file, 'r', encoding='utf-8') as csv_file:
#     reader = csv.DictReader(csv_file)  # Read as a dictionary to access columns by name
#     with open(output_file, 'w', encoding='utf-8') as txt_file:
#         for i, row in enumerate(reader, start=1):  # Enumerate rows, starting at 1
#             if start_row <= i < end_row:  # Write rows within the specified range
#                 txt_file.write(row['sentence_text'] + '\n')

# Input and output file paths
input_file = 'gold_BUG.csv'
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


# Write the selected rows to the output file (simple text format)
with open(output_file, 'w', encoding='utf-8') as txt_file:
    for group, rows in conditions.items():
        for sentence, _, _ in rows:
            txt_file.write(sentence + '\n')

# Write the detailed output file (includes predicted_gender and stereotype)
with open(detailed_output_file, 'w', encoding='utf-8') as detailed_txt_file:
    for group, rows in conditions.items():
        for sentence, gender, stereotype in rows:
            detailed_txt_file.write(f"{sentence}\t{gender}\t{stereotype}\n")
