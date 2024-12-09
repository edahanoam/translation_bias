""" Usage:
    <file-name> --lang=LANG_CODE --model=MODEL --n=BATCH_SIZE [--debug]
"""
from docopt import docopt

from transformers import T5Tokenizer, T5ForConditionalGeneration,pipeline
import pandas as pd

from prepare_data import create_ds_fn, transform_to_fast_align


MODEL_INPUT_FORMAT = ("translate English to {}: {}")
BUG_original_sentence='sentence_text'

languages_dic= {'fr':'French','de':'German'}
models_dic= {'tb':'google-t5/t5-base','nl':'facebook/nllb-200-distilled-1.3B'}


def load_data():
    data=pd.read_csv('gold_BUG.csv',nrows=50)
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



def translation_with_pipeline(df,model,batch_size):

    translator = pipeline("translation", model=model,device=0 )
    results = []
    for i in range(0, len(df), batch_size):
        # Get a batch of inputs
        batch_inputs = df['inputs'].iloc[i:i + batch_size].tolist()

        # Use the pipeline to process the batch
        batch_outputs = translator(batch_inputs)

        # Extract the translations from the batch outputs
        results.extend([output['translation_text'] for output in batch_outputs])

    # Add results back to the DataFrame
    df['translated_text'] = results
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

    # out_bi = args["--bi"] #i am a text file containig the formatted to dast allign text
    # out_ds = args["--ds"] # i think i am a file oin the structure: gender proffession_index sententence proffession
    # align_fn = args["--align"] # i am the fast a allign file
    #

    data = load_data()
    data = transform_to_model_input(data,lang)

    model = models_dic[model_name]

    if model == 'tb':
        df = inference_one_model(data,model,batch_size)
    else:
        df = translation_with_pipeline(data,model,batch_size)

    transform_to_fit_eval(df,f"ds_BUG.txt",f"to_align_{lang}_{model_name}")
