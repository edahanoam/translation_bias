""" Usage:
    <file-name> --in=INPUT_FILE --out=OUTPUT_FILE [--debug]

"""


from docopt import docopt
from pathlib import Path
import spacy
from datasets import load_dataset
from huggingface_hub import login
login()

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


def load_data(ambi=False):
    ds = load_dataset("FBK-MT/gender-bias-PE", "all",split='test')
    print(ds.shape)
    if not ambi:
        data = ds.filter(lambda x: x['dataset'] == 'mtgen_un')
    else:
        data = ds.filter(lambda x: x['dataset'] == 'mtgen_a')
    return data



def find_entities_spacy(text, nlp):
    #text = "Barack Obama and Michelle went to the White House."
    doc = nlp(text)
    human_entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    pronouns = [token.text for token in doc if token.pos_ == "PRON" and token.lower_ in {"he", "she", "they", "him", "her", "them"}]
    human_entities.extend(pronouns)

    #print("Human entities found:", human_entities)
    return human_entities

def find_all_entities(dataset,english_col,model=None):
    nlp = spacy.load("en_core_web_sm")
    if not model:
        dataset = dataset.map(
            lambda row: {"entity": find_entities_spacy(row[english_col],nlp)},
            batched=False
        )

    return dataset


def transform_to_fast_align(dataset, original_text_column, translation_column, out_fn):

    def format_row(row):
        return {"formatted_text": f"{row[original_text_column]} ||| {row[translation_column]}"}

    formatted_lines = dataset.map(format_row, remove_columns=dataset.column_names)

    # Write to file in one go
    with open(out_fn, "w", encoding="utf-8") as f:
        f.write("\n".join(formatted_lines["formatted_text"]))






if __name__ == '__main__':
    # Parse command line arguments
    args = docopt(__doc__)
    in_file = args["--in"]
    out_fn = Path(args["--out"])
    #find_entities()
    data = load_data(False)
    data=find_all_entities(data,english_col="segment")
    print(data['entity'])
    transform_to_fast_align(data, 'segment', 'tgt', 'fast_align.txt')



