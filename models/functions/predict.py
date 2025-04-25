from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    pipeline, 
    logging
)

def predict(test, model, tokenizer, categories):
    """
    Predict the labels for the test set.
    
    Args:
        test (pandas.DataFrame): The test set.
        model (transformers.PreTrainedModel): The loaded model.
        tokenizer (transformers.PreTrainedTokenizer): The loaded tokenizer.
        categories (list): The list of categories.

    Returns:
        y_pred (list): The predicted labels.
    """
    y_pred = []

    transformers.logging.set_verbosity_error()
    
    for i in tqdm(range(len(test)), leave=True):
        prompt = test.iloc[i]["prompt"]
        pipe = pipeline(
                    task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=5,
                    temperature=0.1
                )

        result = pipe(prompt)
        answer = result[0]['generated_text'].split("label:")[-1].strip()

        y_labels = []

        # Determine the predicted category
        for category in categories:
            if category.lower() in answer.replace(r'\n', ' ').lower():
                y_labels.append(category)

        y_pred.append(y_labels)

    return y_pred