# Math Meme Correction Model

This project fine-tunes the **DeepSeek Math 7B Instruct** model to correct common mathematical misconceptions and errors found in viral math memes. The model is trained using **LoRA (Low-Rank Adaptation)** for parameter-efficient tuning.

## Features
- **Automated Correction**: Identifies and corrects incorrect mathematical statements.
- **Instruction-Tuned**: Trained using a dataset of common math misconceptions.
- **Fine-Tuned with LoRA**: Efficiently fine-tuned using low-rank adaptation.
- **Error Rating System**: Provides a fun "sass vs. patience" rating for each correction.

## Installation
Ensure you have Python installed, then install the necessary dependencies:

```bash
pip install datasets transformers torch peft bitsandbytes
```

## Dataset
The dataset consists of incorrect math statements and their corresponding corrections, formatted for instruction tuning. Example:

```json
{
  "incorrect": "8 ÷ 2(2+2) = 1",
  "correction": "The correct answer is 16. The error is in applying PEMDAS incorrectly..."
}
```

## Model Training
The training process involves:
1. **Preprocessing**: Formatting data for instruction tuning.
2. **Tokenization**: Using the DeepSeek tokenizer with padding and truncation.
3. **LoRA Fine-Tuning**: Efficient training with reduced computational cost.
4. **Evaluation**: Validating performance on a test set.

To train the model, run:

```python
trainer.train()
```

### Training Configuration
- **Model**: `deepseek-ai/deepseek-math-7b-instruct`
- **Batch Size**: 4 (with gradient accumulation steps of 2)
- **Learning Rate**: 1e-4
- **Epochs**: 3
- **Evaluation Strategy**: Per epoch

## Model Deployment
After training, save the fine-tuned model and tokenizer:

```python
model.save_pretrained("./math-meme-repair-model")
tokenizer.save_pretrained("./math-meme-repair-model")
```

## Usage
To generate a correction for a new math meme:

```python
def generate_correction(meme, model, tokenizer):
    prompt = f"### Instruction:\nFix this incorrect math meme: {meme}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=256)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return response
```

### Example Usage:
```python
user_meme = "5^2 = 10"
correction = generate_correction(user_meme, model, tokenizer)
print(f"Incorrect: {user_meme}\nCorrection: {correction}")
```

## Fun Feature: Error Rating System
The model assigns a "sass vs. patience" rating for each correction:

```python
import random
sass_level = random.randint(70, 95)
patience_level = 100 - sass_level
print(f"Model Error Rating: {sass_level}% sass, {patience_level}% patience")
```

## Future Improvements
- **Expand Dataset**: Incorporate more complex mathematical misconceptions.
- **Fine-Tune with More Data**: Improve generalization to unseen errors.
- **Deploy as API**: Serve corrections via a web interface.

## License
This project is open-source and licensed under the MIT License.

---

For any questions or contributions, feel free to reach out!

