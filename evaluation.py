""" Usage:
    <file-name> --in=INPUT_FILE --out=OUTPUT_FILE [--debug]

"""


from docopt import docopt
from pathlib import Path
import spacy


def find_entities():
    pass

if __name__ == '__main__':
    # Parse command line arguments
    args = docopt(__doc__)
    in_file = args["--in"]
    out_fn = Path(args["--out"])
    print("you got it")

