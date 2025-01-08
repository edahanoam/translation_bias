""" Usage:
    <file-name> --model=MODEL --n=BATCH_SIZE --out=FILENAME [--debug]
"""
from transformers import pipeline
from docopt import docopt
import pandas as pd


model_dics = {'op':"Helsinki-NLP/opus-mt-en-he"}

def load_data():
    #data=pd.read_csv('/cs/snapless/gabis/noam.dahan1/data_BUG/balanced_BUG.csv')
    #data = data.sample(frac=1)
    data = pd.read_csv("unambi_dataDec8.csv")
    return data.head(300)


def translate_with_pipeline(model_name, df, batch_size):
    pipe = pipeline("translation_en_to_he", model=model_name, device='cuda:0')
    # List to hold all translated texts
    print("finished model loading")
    translated_texts = []

    # Process the DataFrame in batches
    for i in range(0, len(df), batch_size):
        # Extract a batch of sentences
        #batch = df['sentence_text'].iloc[i:i + batch_size].tolist()
        batch = df['segment'].iloc[i:i + batch_size].tolist()

        # Translate the batch
        translations = pipe(batch)

        # Extract translated text and extend the results list
        translated_texts.extend([translation['translation_text'] for translation in translations])
    print("finished one batch")

    # Assign the translations back to the DataFrame (or return a new one if preferred)
    df['translated_text'] = translated_texts

    return df


if __name__ == '__main__':
    print("starting")
    args = docopt(__doc__)
    # model = args["--model"]
    model_name = args["--model"]
    batch_size = int(args["--n"])
    out_file = args["--out"]
    print("processed args")
    df = load_data()
    print("loaded data")

    df = translate_with_pipeline(model_dics[model_name], df, batch_size=batch_size)
    df.to_csv(out_file, index=False)
