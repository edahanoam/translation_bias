""" Usage:
    <file-name> --bi=SEGMENTS_TRANSLATION --ds=DS --out=NAME_NEW_BI[--debug]

"""

from docopt import docopt

def create_new_bi_filtered(bi_fn,ds_ds,out_fn):
    # Define the file paths

    def read_file1(bi_fn):
        """Reads the first file and returns a list of entries with English sentences and their translations."""
        with open(bi_fn, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        entries = {line.split(' ||| ')[0]: line.strip() for line in lines}
        return entries

    def read_file2(ds_ds):
        """Reads the second file and extracts English sentences."""
        with open(ds_ds, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        sentences = {line.split('\t')[2].strip() for line in lines}
        return sentences

    def write_output(matching_entries, output_path):
        """Writes the matching entries to the output file."""
        with open(output_path, 'w', encoding='utf-8') as file:
            for entry in matching_entries:
                file.write(entry + '\n')

    # Read and process both files
    entries_file1 = read_file1(bi_fn)
    sentences_file2 = read_file2(ds_ds)

    # Find matches
    matching_entries = [full_entry for sentence, full_entry in entries_file1.items() if sentence in sentences_file2]

    # Write output
    write_output(matching_entries, out_fn)

    print(f'Matching sentences with translations written to {out_fn}')


if __name__ == '__main__':
    args = docopt(__doc__)

    bi_fn = args["--bi"] #i am a text file containig the formatted to dast allign text
    ds_fn = args["--ds"] # i think i am a file oin the structure: gender proffession_index sententence proffession
    out_fn = args["--out"]
    create_new_bi_filtered(bi_fn,ds_fn,out_fn)