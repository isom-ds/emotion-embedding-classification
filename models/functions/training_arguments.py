from transformers import TrainingArguments

def training_arguments(
    output_dir,
    n_epochs,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=1,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=False,
    lr_scheduler_type="cosine",
    report_to="wandb",
    eval_strategy="steps",
    eval_steps = 0.2
):
    """
    Create a configuration for the training arguments.
    
    Args:
        output_dir (str): The directory to save and repository id.
        n_epochs (int): The number of training epochs.
        per_device_train_batch_size (int): The batch size per device.
        gradient_accumulation_steps (int): The number of gradient accumulation steps.
        gradient_checkpointing (bool): Whether to use gradient checkpointing.
        optim (str): The optimizer to use.
        logging_steps (int): The number of steps between logging.
        learning_rate (float): The learning rate.
        weight_decay (float): The weight decay.
        fp16 (bool): Whether to use FP16.
        bf16 (bool): Whether to use BF16.
        max_grad_norm (float): The maximum gradient norm.
        max_steps (int): The maximum number of steps.
        warmup_ratio (float): The warmup ratio.
        group_by_length (bool): Whether to group by length.
        lr_scheduler_type (str): The learning rate scheduler type.
        report_to (str): The reporting tool.
        eval_strategy (str): The evaluation strategy.
        eval_steps (float): The evaluation steps.
    
    Returns:
        TrainingArguments: The training arguments.
    """

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        optim=optim,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to=report_to,
        eval_strategy=eval_strategy,
        eval_steps eval_steps,
    )
    
    return training_arguments