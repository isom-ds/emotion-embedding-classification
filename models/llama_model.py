class llama_model(nn.Module):
    def __init__(self):
        super(llama_model, self).__init__()
        self.model, self.tokenizer = load_model_tokenizer(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            hf_token="YOUR_HF_TOKEN"
        )

    def predict(self, prompt, max_length=512):
        return self.model.generate(
            self.tokenizer.encode(prompt, return_tensors="pt"),
            max_length=max_length
        )

    def evaluate(self, prompt, target):
        return evaluate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            target=target
        )

    def train(self, train_data, eval_data):
        trainer = SFTTrainer(
            model=model,
            args=training_arguments,
            train_dataset=train_data,
            eval_dataset=eval_data,
            peft_config=peft_config,
            dataset_text_field="prompt",
            tokenizer=tokenizer,
            max_seq_length=512,
            packing=False,
            dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
            }
        )