from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)

def load_model_tokenizer(
    model_name,
    hf_token
):
    """
    Load a model from the Hugging Face model hub.

    Args:
        model_name (str): The name of the model to load.
        hf_token (str): The Hugging Face API token.

    Returns:
        model (transformers.PreTrainedModel): The loaded model.
        tokenizer (transformers.PreTrainedTokenizer): The loaded tokenizer.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="float16",
        quantization_config=bnb_config, 
        token=hf_token
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        token=hf_token
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer