""" Usage:
    <file-name> --in=INPUT_FILE --out=OUTPUT_FILE [--debug]

"""


from docopt import docopt
from pathlib import Path
import spacy
from datasets import load_dataset
from huggingface_hub import login
from main import get_proffession_list,filter_profession,merge_sterio_anti
import pandas as pd
from WinoMTSupport.spacy_support import SpacyPredictor
from WinoMTSupport.gendered_article import GenderedArticlePredictor, \
    get_german_determiners, GERMAN_EXCEPTION, get_french_determiners
from WinoMTSupport.load_alignments import align_bitext_to_ds, get_translated_professions



login()

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

LANGAUGE_PREDICTOR = {
    "es": lambda: SpacyPredictor("es"),
    #"fr": lambda: SpacyPredictor("fr"),
    "it": lambda: SpacyPredictor("it"),
    #"ru": lambda: PymorphPredictor("ru"),
    #"uk": lambda: PymorphPredictor("uk"),
    #"he": lambda: HebrewPredictor(),
    #"ar": lambda: ArabicPredictor(),
    "de": lambda: GenderedArticlePredictor("de", get_german_determiners, GERMAN_EXCEPTION),
    #"cs": lambda: CzechPredictor(),
    #"pl": lambda: MorfeuszPredictor(),
}


def load_data(ambi=False):
    ds = load_dataset("FBK-MT/gender-bias-PE", "all",split='test')
    print(ds.shape)
    if not ambi:
        data = ds.filter(lambda x: x['dataset'] == 'mtgen_un')
    else:
        data = ds.filter(lambda x: x['dataset'] == 'mtgen_a')
    return data


def find_entities_spacy(text, nlp,include_pronouns=False):
    #text = "Barack Obama and Michelle went to the White House."
    doc = nlp(text)
    human_entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    if include_pronouns:
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



def find_all_professions(dataset,english_col,model=None):
    nlp = spacy.load("en_core_web_sm")
    if not model:
        dataset = dataset.map(
            lambda row: {"entity": find_entities_spacy(row[english_col],nlp)},
            batched=False
        )

    return dataset



def transform_to_fast_align(dataset, original_text_column, translation_column, out_fn):
#todo: move to a different file, it is not a part of the pipeline
    def format_row(row):
        return {"formatted_text": f"{row[original_text_column]} ||| {row[translation_column]}"}

    formatted_lines = dataset.map(format_row, remove_columns=dataset.column_names)

    # Write to file in one go
    with open(out_fn, "w", encoding="utf-8") as f:
        f.write("\n".join(formatted_lines["formatted_text"]))





def identify_gender_ttranslations(df, lang):
    gender_predictor = LANGAUGE_PREDICTOR[lang]()
    if lang=='German':
        pass

    if lang=='Italian':
        pass

    if lang=='Spanish':
        pass


def create_ds_fn(data, original_text_column):
    df = data.to_pandas()
    selected_columns = ['gender', 'profession_index','segment', 'profession']
    reordered_df = df[selected_columns]

    # Convert the reordered DataFrame to a list of lists
    ds = reordered_df.values.tolist()
    return ds


if __name__ == '__main__':
    # Parse command line arguments
    #args = docopt(__doc__)
    # in_file = args["--in"]
    # out_fn = Path(args["--out"])

    #bi_fn = args["--bi"] #i am a text file containig the formatted to dast allign text
    #ds_fn = args["--ds"] # i think i am a file oin the structure: gender proffession_index sententence proffession

    #align_fn = args["--align"] # i am the fast a allign file
    #find_entities()
    #data = load_data(True)
    #data=find_all_entities(data,english_col="segment")
    #print(data['entity'])

    data = merge_sterio_anti(pd.read_csv("gold_BUG.csv"),filter_profession(load_data(True)),get_proffession_list())
    print(type(data))
    ds = create_ds_fn(data,'segment')
    print(ds)
    #transform_to_fast_align(data, 'segment', 'tgt', 'fast_align.txt')

    #ds_fn = create_ds_fn(data)
    #ds = [line.strip().split("\t") for line in open(ds_fn, encoding = "utf8")]
    #full_bitext = [line.strip().split(" ||| ")
     #         for line in open(bi_fn, encoding = "utf8")]
    #bitext = align_bitext_to_ds(full_bitext, ds)

    #translated_profs, tgt_inds = get_translated_professions(align_fn, ds, bitext)
    #assert(len(translated_profs) == len(tgt_inds))

