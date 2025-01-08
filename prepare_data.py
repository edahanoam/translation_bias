from datasets import load_dataset
from main import get_proffession_list,filter_profession,merge_sterio_anti
import pandas as pd
import re

def load_data(ambi=False):
    ds = load_dataset("FBK-MT/gender-bias-PE", "all",split='test')
    print(ds.shape)
    if not ambi:
        data = ds.filter(lambda x: x['dataset'] == 'mtgen_un')
    else:
        data = ds.filter(lambda x: x['dataset'] == 'mtgen_a')
    return data


def transform_to_fast_align(dataset, original_text_column, translation_column, out_fn):
    def format_row(row):
        #row[translation_column] = row[original_text_column].str.replace(r'\s*\d+(\.\d+)?$', '', regex=True)
        #row[translation_column] = re.sub(r'\s*\d+(\.\d+)?$', '', row[translation_column])

        return {"formatted_text": f"{row[original_text_column]} ||| {row[translation_column]}"}


    if type(dataset)==pd.DataFrame:
        dataset['formatted_text'] = dataset.apply(format_row, axis=1)
        formatted_lines = [entry['formatted_text'] for entry in dataset['formatted_text']]
        with open(out_fn, "w", encoding="utf-8") as f:
            f.write("\n".join(formatted_lines))

    else:
        formatted_lines = dataset.map(format_row, remove_columns=dataset.column_names)

        # Write to file in one go
        with open(out_fn, "w", encoding="utf-8") as f:
            f.write("\n".join(formatted_lines["formatted_text"]))


def using_italiandata():
    #curently - un-ambig

    data = merge_sterio_anti(pd.read_csv("Data/gold_BUG.csv"), filter_profession(load_data(False)), get_proffession_list())

    filtered_dataset = data.filter(lambda row: None not in row.values())
    df = filtered_dataset.to_pandas()

    # Save the DataFrame to a CSV file
    df.to_csv('unambi_dataDec8.csv', index=False)

    #transform_to_fast_align(filtered_dataset, 'segment', 'suggestion', 'fast_align_Dec8.txt')

    # Splitting the DataFrame based on the 'gender' column
    df_male = df[df['gender'] == 'M']
    df_female = df[df['gender'] == 'F']

    create_ds_fn(df_male, "male_ds_italian_data.txt")
    create_ds_fn(df_female, "female_ds_italian_data.txt")

    transform_to_fast_align(df_male, 'segment', 'suggestion', 'male_bi_italian_data.txt')
    transform_to_fast_align(df_female, 'segment', 'suggestion', 'female_bi_italian_data.txt')

    ### and then needed for the after human manipulation:
    transform_to_fast_align(df_male, 'segment', 'last_translation', 'after_male_bi_italian_data.txt')
    transform_to_fast_align(df_female, 'segment', 'last_translation', 'after_female_bi_italian_data.txt')

    # and now all the process for pro-anti stero
    pro = df[df['stereotype'] == 1]
    anti = df[df['stereotype'] == -1]
    create_ds_fn(pro, "pro_ds_italian_data.txt")
    create_ds_fn(anti, "anti_ds_italian_data.txt")

    transform_to_fast_align(pro, 'segment', 'suggestion', 'pro_bi_italian_data.txt')
    transform_to_fast_align(anti, 'segment', 'suggestion', 'anti_bi_italian_data.txt')

    ### and then needed for the after human manipulation:
    transform_to_fast_align(pro, 'segment', 'last_translation', 'after_pro_bi_italian_data.txt')
    transform_to_fast_align(anti, 'segment', 'last_translation', 'after_anti_bi_italian_data.txt')


def create_ds_fn(data, oringal_text='segment',profession_index = 'profession_index', gender_col = 'gender',out_fn='dsDec8.txt'):
    #df = data.to_pandas()

    selected_columns = [gender_col, profession_index,oringal_text, 'profession']
    reordered_df = data[selected_columns]

    reordered_df[gender_col] = reordered_df[gender_col].replace({'F': 'female', 'M': 'male'})

    # Convert the reordered DataFrame to a list of lists
    ds = reordered_df.values.tolist()

    def save_ds_as_txt(ds, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for row in ds:
                f.write('\t'.join(map(str, row)) + '\n')

    # Save ds
    save_ds_as_txt(ds, out_fn)
    return ds


if __name__ == '__main__':
    using_italiandata()
    # gold_BUG= pd.read_csv("gold_BUG.csv")
    # transform_to_fast_align(gold_BUG,'sentence_text')




