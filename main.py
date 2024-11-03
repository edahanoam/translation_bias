import pandas as pd
from datasets import load_dataset
import evaluate
import re



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


def merge_sterio_anti(df_bug,ds,proffesion_list):
    # first change columns value to much the FBK
    df_bug["predicted gender"] = df_bug["predicted gender"].replace({"Male": "M", "Female": "F"})

    def categorize_text(example):
        # Extract the profession mentioned in the text
        profession_found = None
        for profession in proffesion_list:
            if re.search(rf"\b{profession}\b", example["segment"], re.IGNORECASE):
                profession_found = profession
                break

        # Find the stereotype based on profession and gender
        stereotype = None
        if profession_found:
            match = df_bug[(df_bug['profession'] == profession_found) &
                                   (df_bug['predicted gender'] == example["gender"])]
            if not match.empty:
                stereotype = match.iloc[0]['stereotype']

        # Add extracted information to the example
        example["profession"] = profession_found
        example["stereotype"] = stereotype
        return example

    # Apply the categorize_text function to the dataset
    dataset_texts = ds.map(categorize_text)

    # Function to adjust stereotypes based on opposite gender
    def adjust_stereotype(example):
        if example['stereotype'] is None and example['profession'] is not None:
            # Find the opposite gender
            opposite_gender = 'F' if example['gender'] == 'M' else 'M'
            opposite_match = df_bug[(df_bug['profession'] == example['profession']) &
                                            (df_bug['predicted gender'] == opposite_gender)]
            if not opposite_match.empty:
                # Invert the stereotype value
                example['stereotype'] = -opposite_match.iloc[0]['stereotype']

        return example

    filtered_dataset = dataset_texts.filter(lambda example: example['stereotype'] is not None)

    # Count the rows that don't have None in the 'stereotype'
    count_non_none_stereotype = len(filtered_dataset)

    print("Number of rows without None in the stereotype column:", count_non_none_stereotype)

    # Second pass to adjust None stereotypes
    dataset_texts = dataset_texts.map(adjust_stereotype)
    # Filter the dataset to remove rows where 'stereotype' is None
    filtered_dataset = dataset_texts.filter(lambda example: example['stereotype'] is not None)

    # Count the rows that don't have None in the 'stereotype'
    count_non_none_stereotype = len(filtered_dataset)

    print("Number of rows without None in the stereotype column:", count_non_none_stereotype)



def filter_profession(ds):
    prof_list=get_proffession_list()
    print(len(ds))
    filtered_data = ds.filter(lambda x: any(word in x['segment'] for word in prof_list))
    print(len(filtered_data))
    return filtered_data



if __name__ == '__main__':
    #print("Columns:", data.column_names)
    filter_profession(load_data(False))
    #calc_all_options()
    merge_sterio_anti(pd.read_csv("gold_BUG.csv"),filter_profession(load_data(False)),get_proffession_list())


    #sanity check: reproduce the bleu score in paper
    # suggestions = data['suggestion']
    # references = data['tgt']
    # refrences_array = [[s] for s in references]
    # all_score= bleu.compute(predictions=suggestions, references=refrences_array)
    # print(all_score['bleu'])



