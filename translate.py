from transformers import T5Tokenizer, T5ForConditionalGeneration



if __name__ == '__main__':
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)

    input = "My name is Azeem and I live in India"

    # You can also use "translate English to French" and "translate English to Romanian"
    input_ids = tokenizer("translate English to Italian: " + input, return_tensors="pt").input_ids  # Batch size 1

    outputs = model.generate(input_ids)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(decoded)

