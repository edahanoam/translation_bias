import pandas as pd
from datasets import load_dataset
#import evaluate
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
    """calc the difference between male and female sentences"""
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
    num_profs= len(df.profession.unique())
    df= pd.read_csv("full_BUG.csv")

    prof_list=df.profession.unique()
    print(num_profs==len(prof_list))
    print(prof_list)
    return prof_list


def merge_sterio_anti(df_bug,ds,proffesion_list):
    """This function merge the bug info about proffessions and steriotypes info the FBK"""
    # first change columns value to much the FBK
    df_bug["predicted gender"] = df_bug["predicted gender"].replace({"Male": "M", "Female": "F"})

    def categorize_text(example):
        """this function looks for proffessions in the text and """
        # Extract the profession mentioned in the text
        profession_found = None
        profession_index = None

        for profession in proffesion_list:
            match =re.search(rf"\b{profession}\b", example["segment"], re.IGNORECASE)
            if match:
                profession_found = profession
                profession_index = int(len(example["segment"][:match.start()].split()))
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
        example["profession_index"] = profession_index
        example["profession_index"] =example["profession_index"]
        return example

    # Apply the categorize_text function to the dataset
    dataset_texts = ds.map(categorize_text)

    # Function to adjust stereotypes based on opposite gender
    def find_opposite_gender(example):
        """this function goes over unmatched professions and see
        if they are included in the bug dataset for the opposite gender"""
        if example['stereotype'] is None and example['profession'] is not None:
            # Find the opposite gender
            opposite_gender = 'F' if example['gender'] == 'M' else 'M'
            opposite_match = df_bug[(df_bug['profession'] == example['profession']) &
                                            (df_bug['predicted gender'] == opposite_gender)]
            if not opposite_match.empty:
                # Invert the stereotype value
                example['stereotype'] = -opposite_match.iloc[0]['stereotype']

        return example

    # Second pass to adjust None stereotypes
    dataset_texts = dataset_texts.map(find_opposite_gender)
    return dataset_texts


def filter_profession(ds):
    prof_list=get_proffession_list()
    print(len(ds))
    filtered_data = ds.filter(lambda x: any(word in x['segment'] for word in prof_list))
    print(len(filtered_data))
    return filtered_data


def calc_bleu_dif_stereotype(ds):
    """calculate the differences between stereotypical and anti stereotypical sentences"""
    bleu = evaluate.load("bleu")
    anti = ds.filter(lambda x: x['stereotype'] == -1.0)
    pro = ds.filter(lambda x: x['stereotype'] == 1.0)
    post_edits_anti = anti['last_translation']
    suggestions_anti = anti['suggestion']
    references_anti = anti['tgt']
    refrences_array_anti = [[s] for s in references_anti]
    post_edits_pro = pro['last_translation']
    suggestions_pro = pro['suggestion']
    references_pro = pro['tgt']
    refrences_array_pro = [[s] for s in references_pro]
    antibleu,pro_bleu,dif = calc_dif(bleu, suggestions_pro, refrences_array_pro, suggestions_anti, refrences_array_anti)
    print(f"anti bleu {antibleu} pro bleu {pro_bleu} dif {dif}")
    antibleu_after,pro_bleu_after,dif_after= calc_dif(bleu, post_edits_pro, refrences_array_pro, post_edits_anti, refrences_array_anti)
    print(f"anti bleu {antibleu_after} pro bleu {pro_bleu_after} dif {dif_after}")


if __name__ == '__main__':
    #print("Columns:", data.column_names)
    #filter_profession(load_data(False))
    #calc_all_options()
    # ds = merge_sterio_anti(pd.read_csv("gold_BUG.csv"),filter_profession(load_data(False)),get_proffession_list())
    # calc_bleu_dif_stereotype(ds)

    print(get_proffession_list())
    #sanity check: reproduce the bleu score in paper
    # suggestions = data['suggestion']
    # references = data['tgt']
    # refrences_array = [[s] for s in references]
    # all_score= bleu.compute(predictions=suggestions, references=refrences_array)
    # print(all_score['bleu'])



