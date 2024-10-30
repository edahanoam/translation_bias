import pandas as pd
from datasets import load_dataset
from huggingface_hub import login


if __name__ == '__main__':


    ds = load_dataset("FBK-MT/gender-bias-PE", "all")
    print(ds.shape)
