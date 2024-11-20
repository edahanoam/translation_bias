""" Usage:
    <file-name> --in=INPUT_FILE --out=OUTPUT_FILE [--debug]

"""


from docopt import docopt
from pathlib import Path
import spacy
from datasets import load_dataset
from huggingface_hub import login
login()



def load_data(ambi=False):
    ds = load_dataset("FBK-MT/gender-bias-PE", "all",split='test')
    print(ds.shape)
    if not ambi:
        data = ds.filter(lambda x: x['dataset'] == 'mtgen_un')
    else:
        data = ds.filter(lambda x: x['dataset'] == 'mtgen_a')
    return data


def find_entities(text):
    nlp = spacy.load("en_core_web_sm")
    #text = "Barack Obama and Michelle went to the White House."
    doc = nlp(text)
    human_entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    #print("Human entities found:", human_entities)
    return human_entities

def find_all_entities(dataset,english_col):
    #todo: this is very slow consider using something else
    dataset = dataset.map(
        lambda row: {"entity": find_entities(row[english_col])},
        batched=False
    )
    return dataset



if __name__ == '__main__':
    # Parse command line arguments
    args = docopt(__doc__)
    in_file = args["--in"]
    out_fn = Path(args["--out"])
    #find_entities()
    data = load_data(True)
    data=find_all_entities(data,english_col="segment")
    print(data['entity'])

