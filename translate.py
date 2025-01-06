""" Usage:
    <file-name> --lang=LANG_CODE --model=MODEL --n=BATCH_SIZE --out=FILENAME [--debug]
"""
from docopt import docopt

from transformers import T5Tokenizer, T5ForConditionalGeneration,pipeline
import pandas as pd
import torch
from prepare_data import create_ds_fn, transform_to_fast_align
from torch.utils.data import DataLoader


MODEL_INPUT_FORMAT = ("translate English to {}: {}")
BUG_original_sentence='sentence_text'

languages_dic= {'fr':'French','de':'German','es':'Spanish','he':'Hebrew'}
models_dic= {'tb':'google-t5/t5-base','nl':'facebook/nllb-200-distilled-1.3B','op': 'Helsinki-NLP/opus-mt-en-he '}


def load_data():
    data=pd.read_csv('gold_BUG.csv',nrows=50)
    #data=pd.read_csv('gold_BUG.csv')

    return data


def transform_to_model_input(df,lang):
    df['inputs'] = None
    for i in range(len(df)):
        df['inputs'][i] = MODEL_INPUT_FORMAT.format(languages_dic[lang],df[BUG_original_sentence][i])
    return df


def inference_one_model(df,model,batch_size):
    # print("yo")
    tokenizer = T5Tokenizer.from_pretrained(model)

    model = T5ForConditionalGeneration.from_pretrained(model, return_dict=True)

    # Process the DataFrame in batches
    # Process the DataFrame in batches
    results = []
    for i in range(0, len(df), batch_size):
        # Get a batch of inputs
        batch_inputs = df['inputs'].iloc[i:i + batch_size].tolist()

        # Tokenize without adding a prefix
        tokenized_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True)

        # Generate outputs
        outputs = model.generate(tokenized_inputs.input_ids)

        # Decode outputs and collect results
        decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        results.extend(decoded_outputs)

    # Add results back to the DataFrame
    df['translated_text'] = results
    print(df.head(10))
    return df



# def translation_with_pipeline(df,model,batch_size,lang):
#
#     translator = pipeline("translation", model=model,device=0,src_lang='en', tgt_lang=lang)
#     results = []
#     for i in range(0, len(df), batch_size):
#         # Get a batch of inputs
#         batch_inputs = df['sentence_text'].iloc[i:i + batch_size].tolist()
#
#         # Add the target language code to each sentence
#         formatted_batch = [f">>{lang}<< {sentence}" for sentence in batch_inputs]
#
#         # Translate the batch
#         translations = translator(formatted_batch)
#
#         # Extract the translation text
#         results.extend([translation['translation_text'] for translation in translations])
#
#         # Use the pipeline to process the batch
#         batch_outputs = translator(batch_inputs)
#
#         # Extract the translations from the batch outputs
#         results.extend([output['translation_text'] for output in batch_outputs])
#
#     # Add results back to the DataFrame
#     df['translated_text'] = results
#     print(df.head(10))
#     return df
#


def translation_with_pipeline(df, model, batch_size, lang):
    # Initialize the translation pipeline with specified model and device
    translator = pipeline("translation", model=model, device=0, src_lang='en', tgt_lang=lang)

    results = []
    for i in range(0, len(df), batch_size):
        # Get a batch of inputs
        batch_inputs = df['sentence_text'].iloc[i:i + batch_size].tolist()

        # Add the target language code to each sentence (if your model needs this)
        formatted_batch = [f">>{lang}<< {sentence}" for sentence in batch_inputs]

        # Translate the batch
        translations = translator(formatted_batch)

        # Extract the translation text and add to results
        results.extend([translation['translation_text'] for translation in translations])

    # Ensure results list length matches DataFrame's length before assignment
    if len(results) == len(df):
        df['translated_text'] = results
    else:
        print("Error: The number of results does not match the number of DataFrame rows.")

    print(df.head(10))
    return df


def transform_to_fit_eval(df,out_ds,out_bi):
    create_ds_fn(df,'sentence_text','profession_first_index', 'predicted gender',out_ds)
    transform_to_fast_align(df, 'sentence_text', 'translated_text', out_bi)





if __name__ == '__main__':
    args = docopt(__doc__)
    # model = args["--model"]
    lang = args["--lang"] # code for language
    model_name = args["--model"]
    batch_size = int(args["--n"])
    out_file = args["--out"]

    # out_bi = args["--bi"] #i am a text file containig the formatted to dast allign text
    # out_ds = args["--ds"] # i think i am a file oin the structure: gender proffession_index sententence proffession
    # align_fn = args["--align"] # i am the fast a allign file
    #

    data = load_data()
    data = transform_to_model_input(data,lang)

    model = models_dic[model_name]

    if model_name == 'tb':
        df = inference_one_model(data,model,batch_size)
    else:
        df = translation_with_pipeline(data,model,batch_size,lang)

    df.to_csv(out_file)

    #transform_to_fit_eval(df,f"ds_BUG.txt",f"to_align_{lang}_{model_name}")
