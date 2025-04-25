import bitsandbytes as bnb
from peft import LoraConfig

def peft_config(
    model,
    lora_alpha=16,
    lora_dropout=0,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
):
    """
    Create a configuration for the Lora model.
    
    Args:
        model (transformers.PreTrainedModel): The model.
        lora_alpha (int): The alpha value for Lora.
        lora_dropout (float): The dropout rate for Lora.
        r (int): The number of bits to quantize to.
        bias (str): The bias type.
        task_type (str): The task type.
        
    Returns:
        LoraConfig: The Lora configuration.
    """

    def find_all_linear_names(model):
        """
        Find all the names of the linear layers in the model.

        Args:
            model (transformers.PreTrainedModel): The model.

        Returns:
            list: The list of names of the linear layers.
        """
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names:  # needed for 16 bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=r,
        bias=bias,
        task_type=task_type,
        target_modules=find_all_linear_names(model)
    )

    return peft_config