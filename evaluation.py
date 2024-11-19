""" Usage:
    <file-name> --in=INPUT_FILE --out=OUTPUT_FILE [--debug]

"""


from docopt import docopt
from pathlib import Path
import spacy


def find_entities():
    nlp = spacy.load("en_core_web_sm")
    text = "Barack Obama and Michelle went to the White House."
    doc = nlp(text)
    human_entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    print("Human entities found:", human_entities)


if __name__ == '__main__':
    # Parse command line arguments
    args = docopt(__doc__)
    in_file = args["--in"]
    out_fn = Path(args["--out"])
    find_entities()

