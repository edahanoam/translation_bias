import pandas as pd
from datasets import load_dataset
import evaluate


def load_data(ambi=False):
    ds = load_dataset("FBK-MT/gender-bias-PE", "all",split='test')
    print(ds.shape)
    if not ambi:
        data = ds.filter(lambda x: x['dataset'] == 'mtgen_un')
    else:
        data = ds.filter(lambda x: x['dataset'] == 'mtgen_a')
    return data


def get_relevant_cols(ds,Female=True):

    # original = ds['segment']
    # suggestion = ds['suggestion']
    # post_edits = ds['last_translation']
    # references= ds['tgt']
    if Female:
        filtered_dataset = ds.filter(lambda row: row['gender'] == 'F')
    else:
        filtered_dataset = ds.filter(lambda row: row['gender'] == 'M')

    post_edits = filtered_dataset['last_translation']
    suggestions = filtered_dataset['suggestion']
    references = filtered_dataset['tgt']
    refrences_array = [[s] for s in references]


    return suggestions,post_edits,refrences_array


def calc_dif(bleu,suggestions_male,references_male,suggestions_female,references_female):
    male_score=bleu.compute(predictions=suggestions_male, references=references_male)
    print(male_score['bleu'])
    female_score=bleu.compute(predictions=suggestions_female, references=references_female)
    print(female_score['bleu'])
    print(male_score['bleu']-female_score['bleu'])



if __name__ == '__main__':
    data = load_data(False)
    #print("Columns:", data.column_names)

    suggestions_female, post_edits_female, references_female = get_relevant_cols(data)
    suggestions_male, post_edits_male, references_male = get_relevant_cols(data, False)

    bleu = evaluate.load("bleu")

    #calc_dif(bleu,suggestions_male,references_male,suggestions_female,references_female)

    #sanity check: reproduce the bleu score in paper
    suggestions = data['suggestion']
    references = data['tgt']
    refrences_array = [[s] for s in references]
    all_score= bleu.compute(predictions=suggestions, references=refrences_array)
    print(all_score['bleu'])




