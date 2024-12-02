from datasets import load_dataset
from main import get_proffession_list,filter_profession,merge_sterio_anti
import pandas as pd

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
        return {"formatted_text": f"{row[original_text_column]} ||| {row[translation_column]}"}

    formatted_lines = dataset.map(format_row, remove_columns=dataset.column_names)

    # Write to file in one go
    with open(out_fn, "w", encoding="utf-8") as f:
        f.write("\n".join(formatted_lines["formatted_text"]))


if __name__ == '__main__':
    #curently - un-ambig
    data = merge_sterio_anti(pd.read_csv("gold_BUG.csv"),filter_profession(load_data(False)),get_proffession_list())
    filtered_dataset = data.filter(lambda row: None not in row.values())
    df = filtered_dataset.to_pandas()

    # Save the DataFrame to a CSV file
    df.to_csv('unambi_data.csv', index=False)

    transform_to_fast_align(filtered_dataset, 'segment', 'tgt', 'fast_align_unamb_fullprofs.txt')


