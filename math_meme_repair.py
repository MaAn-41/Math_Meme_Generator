import os
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Prepare the dataset
math_meme_data = [
    {
        "incorrect": "8 ÷ 2(2+2) = 1",
        "correction": "The correct answer is 16. The error is in applying PEMDAS incorrectly. After calculating the parentheses (2+2=4), we have 8 ÷ 2(4). Following order of operations, we calculate from left to right for division and multiplication: 8 ÷ 2 = 4, then 4 × 4 = 16."
    },
    {
        "incorrect": "1 + 1 × 0 + 1 = 1",
        "correction": "The correct answer is 2. The error is forgetting that multiplication has precedence over addition. We calculate 1 × 0 = 0 first, then add: 1 + 0 + 1 = 2."
    },
    {
        "incorrect": "5^2 = 10",
        "correction": "The correct answer is 25. The error is misunderstanding what the exponent (^) means. 5^2 means 5 × 5 = 25, not 5 × 2 = 10."
    },
    {
        "incorrect": "-5^2 = -25",
        "correction": "The correct answer is -25. Many get confused thinking it's 25, but the negative sign applies to the number first, then we square it: -(5^2) = -(25) = -25."
    },
    {
        "incorrect": "0.9999... = 0.9",
        "correction": "The correct answer is 1. The repeating decimal 0.9999... is exactly equal to 1, not 0.9. This can be proven algebraically by letting x = 0.9999..., then 10x = 9.9999..., subtract to get 9x = 9, thus x = 1."
    },
    {
        "incorrect": "1/2 + 1/3 = 2/5",
        "correction": "The correct answer is 5/6. The error is adding numerators and denominators directly. Instead, find a common denominator: 1/2 = 3/6 and 1/3 = 2/6, so 1/2 + 1/3 = 3/6 + 2/6 = 5/6."
    },
    {
        "incorrect": "√(a² + b²) = a + b",
        "correction": "This is incorrect. The square root of a sum is not the sum of the square roots. √(a² + b²) represents the hypotenuse of a right triangle with sides a and b (Pythagorean theorem), and is generally not equal to a + b."
    },
    {
        "incorrect": "log(a + b) = log(a) + log(b)",
        "correction": "This is incorrect. The logarithm of a sum is not the sum of logarithms. The correct identity is log(a × b) = log(a) + log(b)."
    },
    {
        "incorrect": "π = 22/7",
        "correction": "π is approximately 3.14159... While 22/7 = 3.14285... is a common approximation, it's not exactly equal to π. π is an irrational number that cannot be expressed as a simple fraction."
    },
    {
        "incorrect": "(a+b)² = a² + b²",
        "correction": "The correct formula is (a+b)² = a² + 2ab + b². The error is forgetting the middle term. When squaring a binomial, we get the square of the first term, plus twice the product of both terms, plus the square of the second term."
    },
    {
        "incorrect": "sin(α+β) = sin(α) + sin(β)",
        "correction": "This is incorrect. The correct formula is sin(α+β) = sin(α)cos(β) + cos(α)sin(β). Trigonometric functions don't distribute over addition."
    },
    {
        "incorrect": "0/0 = 1",
        "correction": "0/0 is undefined, not 1. Division by zero is undefined in standard arithmetic, and 0/0 is an indeterminate form that requires limit analysis in calculus contexts."
    },
    {
        "incorrect": "√(-1) = -1",
        "correction": "√(-1) = i, not -1. The square root of a negative number is imaginary. i is defined such that i² = -1, so √(-1) = i."
    },
    {
        "incorrect": "0.1 + 0.2 = 0.3",
        "correction": "While this is mathematically correct, computers typically store this as 0.30000000000000004 due to floating-point precision issues. The error is assuming computer calculations match exact decimal arithmetic."
    },
    {
        "incorrect": "ln(e^x) = x + C",
        "correction": "The correct answer is ln(e^x) = x, no constant needed. The natural logarithm and exponential function are inverse operations, so they cancel each other out exactly."
    },
    {
        "incorrect": "n ÷ 0 = 0",
        "correction": "Division by zero is undefined, not zero. You cannot divide any number by zero within the real number system."
    },
    {
        "incorrect": "1 = 0.999...",
        "correction": "This is actually correct! 0.999... (with the 9s repeating infinitely) is exactly equal to 1. This can be proven algebraically by setting x = 0.999..., then 10x = 9.999..., subtract to get 9x = 9, thus x = 1."
    },
    {
        "incorrect": "x/0 = ∞",
        "correction": "x/0 is undefined, not infinity. While we approach infinity as the denominator approaches zero, division by exactly zero is undefined in standard arithmetic."
    },
    {
        "incorrect": "sin(30°) = 1/3",
        "correction": "The correct value is sin(30°) = 1/2. The error is confusing sin(30°) with other trigonometric values. For a 30-60-90 triangle, sin(30°) = 1/2, cos(30°) = √3/2, and tan(30°) = 1/√3."
    },
    {
        "incorrect": "The derivative of x² is 2x²",
        "correction": "The correct derivative of x² is 2x, not 2x². The power rule for derivatives states that the derivative of x^n is n·x^(n-1). So for x², the derivative is 2·x^(2-1) = 2x."
    }
]

# Format the data as instruction tuning examples
formatted_data = []
for item in math_meme_data:
    formatted_data.append({
        "input": f"Fix this incorrect math meme: {item['incorrect']}",
        "output": item["correction"]
    })

# Create HuggingFace dataset
dataset = Dataset.from_list(formatted_data)

# Split into training and validation sets (90% train, 10% validation)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Load the Deepseek model and tokenizer
model_name = "deepseek-ai/deepseek-math-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# We'll use a data collator to handle padding dynamically during batch creation
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Modified tokenization function using pad_to_multiple_of and truncation
def process_function(examples):
    # Format inputs with instruction template
    inputs = [f"### Instruction:\nFix this incorrect math meme: {ex}\n\n### Response:\n" for ex in examples["input"]]

    # Format model inputs (prompts)
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Format labels (responses)
    labels = tokenizer(
        examples["output"],
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Create labels with -100 for prompt tokens (we don't want to compute loss on them)
    input_lengths = [len(tokenizer.encode(inp)) for inp in inputs]
    for i in range(len(examples["input"])):
        # Replace prompt token ids with -100
        model_inputs["labels"] = labels["input_ids"].clone()

    return model_inputs

# Apply tokenization to get batched tensors
tokenized_datasets = {
    "train": Dataset.from_dict({
        "input": dataset["train"]["input"],
        "output": dataset["train"]["output"]
    }).map(process_function, batched=True),

    "test": Dataset.from_dict({
        "input": dataset["test"]["input"],
        "output": dataset["test"]["output"]
    }).map(process_function, batched=True)
}

# Create a simpler dataset format for fine-tuning
class MathMemeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.input_ids = dataset["input_ids"]
        self.attention_mask = dataset["attention_mask"]
        self.labels = dataset["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

# Create PyTorch datasets
train_dataset = MathMemeDataset(tokenized_datasets["train"])
eval_dataset = MathMemeDataset(tokenized_datasets["test"])

# Set up LoRA configuration for parameter-efficient fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Prepare the model for training with LoRA
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=1e-4,
    weight_decay=0.01,
    fp16=True,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="none",  # Disable reporting to avoid dependencies
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./math-meme-repair-model")
tokenizer.save_pretrained("./math-meme-repair-model")

# Function to test the model on new examples
def generate_correction(meme, model, tokenizer):
    prompt = f"### Instruction:\nFix this incorrect math meme: {meme}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    # Decode and clean up the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

user_meme = input("Enter an incorrect math meme: ")
correction = generate_correction(user_meme, model, tokenizer)
print(f"Incorrect: {user_meme}")
print(f"\nCorrection: {correction}\n")
print("----------------------------\n")

# Fun error rating
import random
sass_level = random.randint(70, 95)
patience_level = 100 - sass_level
print(f"\nModel Error Rating: {sass_level}% sass, {patience_level}% patience")
