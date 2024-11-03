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


def get_relevant_cols(ds,Female=True,language=None,prof_level=None):
    if prof_level:
        ds = ds.filter(lambda row: row['user_type'] == prof_level)
    if language:
        ds = ds.filter(lambda row: row['lang'] == language)

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
    female_score=bleu.compute(predictions=suggestions_female, references=references_female)
    return male_score['bleu'],female_score['bleu'],male_score['bleu']-female_score['bleu']

def calc_all_options():
    #abmigous
    bleu = evaluate.load("bleu")
    data = load_data(True)
    results=[]
    all_options_abmi= [['it','professional'],['es','professional'],['de','professional'],['it','student']]
    for option in all_options_abmi:
        suggestions_female, post_edits_female, references_female = get_relevant_cols(data, True, option[0],option[1] )
        suggestions_male, post_edits_male, references_male = get_relevant_cols(data, False,  option[0],option[1])
        male,female,dif= calc_dif(bleu,suggestions_male,references_male,suggestions_female,references_female)
        post_male,post_female, post_dif=calc_dif(bleu,post_edits_male,references_male,post_edits_female,references_female)
        results.append({'option':option,'male_bleu':male,'female_bleu':female,'dif':dif,'post_male':post_male,'post_female':post_female,'post_dif':post_dif })

    #unabmigous
    data = load_data(False)
    suggestions_female, post_edits_female, references_female = get_relevant_cols(data, True)
    suggestions_male, post_edits_male, references_male = get_relevant_cols(data, False)
    male, female, dif = calc_dif(bleu, suggestions_male, references_male, suggestions_female, references_female)
    post_male, post_female, post_dif = calc_dif(bleu, post_edits_male, references_male, post_edits_female,
                                                references_female)


    results.append({'option': ['it','not-ambi'], 'male_bleu': male, 'female_bleu': female, 'dif': dif, 'post_male': post_male,
                'post_female': post_female, 'post_dif': post_dif})

    df = pd.DataFrame(results)
    df.to_csv("results.csv")
    print(results)



def get_proffession_list():
    df= pd.read_csv("gold_BUG.csv")
    prof_list=df.profession.unique()
    return prof_list


def filter_profession(ds):
    prof_list=get_proffession_list()
    print(len(ds))
    filtered_data = ds.filter(lambda x: any(word in x['segment'] for word in prof_list))
    print(len(filtered_data))


if __name__ == '__main__':
    #print("Columns:", data.column_names)
    filter_profession(load_data(False))
    #calc_all_options()



    #sanity check: reproduce the bleu score in paper
    # suggestions = data['suggestion']
    # references = data['tgt']
    # refrences_array = [[s] for s in references]
    # all_score= bleu.compute(predictions=suggestions, references=refrences_array)
    # print(all_score['bleu'])



