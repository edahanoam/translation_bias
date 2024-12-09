from docopt import docopt


from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd

def load_data():
    data=pd.read_csv('gold_BUG.csv')
    return data


def transform_to_model_input(data):
    pass



def inference_one_model(data, model,lang,batch_size):
    print("yo")
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)

    input = "My name is Azeem and I live in India"

    # You can also use "translate English to French" and "translate English to Romanian"
    input_ids = tokenizer("translate English to French: " + input, return_tensors="pt").input_ids  # Batch size 1

    outputs = model.generate(input_ids)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(decoded)


def transform_to_fit_eval(data,):
    pass




if __name__ == '__main__':
    args = docopt(__doc__)
    model = args["--model"]
    lang = args["--lang"] # code for language
    batch_size = args["--n"]
    out_bi = args["--bi"] #i am a text file containig the formatted to dast allign text
    out_ds = args["--ds"] # i think i am a file oin the structure: gender proffession_index sententence proffession
    align_fn = args["--align"] # i am the fast a allign file


    data = load_data()