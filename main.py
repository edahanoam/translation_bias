import pandas as pd
from datasets import load_dataset
from huggingface_hub import login


def load_data(ambi=False):
    ds = load_dataset("FBK-MT/gender-bias-PE", "all",split='test')
    print(ds.shape)
    if not ambi:
        data = ds.filter(lambda x: x['dataset'] == 'mtgen_un')
    else:
        data = ds.filter(lambda x: x['dataset'] == 'mtgen_a')
    return data

def get_suggestions_and_edits(ds):
    suggestion = ds['suggestion']
    post_edits = ds['last_translation']
    return suggestion,post_edits


if __name__ == '__main__':
    data = load_data(False)
    #print("Columns:", data.column_names)

    suggestion, post_edits = get_suggestions_and_edits(data)



