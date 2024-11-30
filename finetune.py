from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from tqdm import tqdm

def train_model():
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Phi-3-mini-4k-instruct",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map="auto",
    )

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}
### Response:
{}"""

    EOS_TOKEN = tokenizer.eos_token  # 必须添加 EOS_TOKEN

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        response = examples["response"]
        texts = []
        for instruction, response in zip(instructions, response):
            text = alpaca_prompt.format(instruction, response) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    dataset = load_dataset("bitext/Bitext-retail-banking-llm-chatbot-training-dataset", split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    # Wrap training to yield progress
    max_steps = trainer.args.max_steps
    for step in tqdm(range(max_steps), desc="Training Progress"):
        trainer_stats = trainer.train_step()
        yield step + 1, max_steps

    # Save the model
    model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
