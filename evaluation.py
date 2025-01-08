""" Usage:
    <file-name> --bi=SEGMENTS_TRANSLATION --align=ALIGN_FILE --ds=DSFILETXT --lang=LANG_CODE --out_fn=RESULTS_FILENAME [--debug]

"""
from tabnanny import process_tokens

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
from WinoMTSupport.load_alignments import align_bitext_to_ds, get_translated_professions, output_predictions
from WinoMTSupport.evaluate import evaluate_bias
from tqdm import tqdm
from WinoMTSupport.util import GENDER, SPACY_GENDER_TYPES
from itertools import islice
from collections import Counter


#login()



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
    nlp = spacy.load("en_core_web_lg")
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




def create_ds_fn(data, out_ds):
    selected_columns = ['gender', 'profession_index','segment', 'profession']
    reordered_df = data[selected_columns]

    # Convert the reordered DataFrame to a list of lists
    ds = reordered_df.values.tolist()

    def save_ds_as_txt(ds, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for row in ds:
                f.write('\t'.join(map(str, row)) + '\n')

    # Save ds
    save_ds_as_txt(ds, out_ds)

    return ds



def predict_gender(word, lang='it'):
    nlp = spacy.load(f"{lang}_core_news_lg", disable=["parser", "ner"])  # Load the model for the specified language

    doc = nlp(word)
    gender_set = set()  # Use a set to store unique genders
    print("one")
    observed_genders = []
    for token in doc:
        #print(f"Token: {token.text}, POS: {token.pos_}, Tags: {token.tag_}, Morph: {token.morph}")
        # Special case handling for Italian
        if (lang == "it") and (token.text.startswith("dell'")):
            observed_genders.append('Masc')
        else:
            # Extract gender, ensure it's not a list
            gender_info = token.morph.get("Gender")
            print(gender_info)
            if gender_info:
                observed_genders.append(gender_info[0])

    if not observed_genders:
        return GENDER.unknown

    return SPACY_GENDER_TYPES.get(Counter(observed_genders).most_common(1)[0][0],0)







def load_ds_from_txt(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        ds = [line.strip().split('\t') for line in f]
    return ds


def for_the_italians(bi_fn,align_fn, ds_fn):
    data = pd.read_csv('unambi_data.csv')

    #print(type(data))
    load_ds_from_txt(ds_fn)
    ds = create_ds_fn(data,'ds.txt')
    print(ds)


    full_bitext = [line.strip().split(" ||| ") for line in open(bi_fn, encoding = "utf8")]
    bitext = align_bitext_to_ds(full_bitext, ds)

    translated_profs, tgt_inds, alignment_pairs = get_translated_professions(align_fn, ds, bitext)

    # Output the alignment pairs
    for pair in alignment_pairs:
        print(pair)

    assert(len(translated_profs) == len(tgt_inds))







if __name__ == '__main__':
    # Parse command line arguments
    args = docopt(__doc__)

    bi_fn = args["--bi"] #i am a text file containig the formatted to dast allign text
    ds_fn = args["--ds"] # i think i am a file oin the structure: gender proffession_index sententence proffession

    align_fn = args["--align"] # i am the fast a allign file
    ds = [line.strip().split("\t") for line in open(ds_fn, encoding = "utf8")]
    lang = args["--lang"] # code for language
    out_fn = args["--out_fn"] # code for language

    print(f"anti for {lang} Dec 5 Italian addition")
    full_bitext = [line.strip().split(" ||| ") for line in open(bi_fn, encoding = "utf8")]
    bitext = align_bitext_to_ds(full_bitext, ds)

    translated_profs, tgt_inds, alignment_pairs = get_translated_professions(align_fn, ds, bitext)
    # Output the alignment pairs
    for pair in alignment_pairs:
        print(pair)

    assert(len(translated_profs) == len(tgt_inds))

    #gender_predictor = LANGAUGE_PREDICTOR[lang]()

    target_sentences = [tgt_sent for (ind, (src_sent, tgt_sent)) in bitext]


    gender_predictions = [predict_gender(prof,lang)
                          for prof, translated_sent, entity_index, ds_entry
                          in tqdm(zip(translated_profs,
                                      target_sentences,
                                      map(lambda ls:min(ls, default = -1), tgt_inds),
                                      ds))]

    print(gender_predictions)


    d = evaluate_bias(ds, gender_predictions)

    with open(out_fn, "w", encoding="utf-8") as file:
        file.write(d)




