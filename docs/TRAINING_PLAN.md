# Neural Network Training Plan for Conan Cases

## ✅ Data Ready: 500 Cases

### Quality Check Results:
- ✅ **500 cases generated** (C001-C500)
- ✅ **50 random cases spot-checked**: 100% pass rate
- ✅ **All cases validated**: No structural errors
- ✅ **Ready for training**

---

## 🎯 Training Objectives

### What We Want the Model to Learn:
1. **Contradiction Detection** - Identify logical inconsistencies in evidence
2. **Deception Recognition** - Spot when physical evidence is manipulated
3. **Mechanism Reasoning** - Understand how tricks work (ice melting, tape illusion, etc.)
4. **False Narrative Rejection** - Distinguish plausible lies from truth
5. **Evidence Weighting** - Prioritize diagnostic evidence over misdirection

---

## 📊 Dataset Structure

### Input Format Options:

#### Option A: Evidence → Solution (Simple)
```python
Input: List of 6 evidence items
Output: Hidden truth explanation
```

#### Option B: Evidence + Question → Answer (QA Format)
```python
Input: Evidence + Detective question
Output: Reasoning + Answer
```

#### Option C: Full Reasoning Chain (Complex)
```python
Input: Evidence + Contradictions + Questions
Output: Step-by-step reasoning → Solution
```

**Recommendation: Start with Option B (QA Format)**

---

## 🔧 Data Preparation

### Step 1: Extract Training Data

Create a script to convert JSON cases into training format:

```python
# training_data_extractor.py
import json
from pathlib import Path

def extract_training_data():
    """Extract training examples from all cases."""
    cases_dir = Path("detective_engine/cases")
    training_data = []
    
    for case_file in sorted(cases_dir.glob("C*.json")):
        with open(case_file) as f:
            data = json.load(f)
        
        case = data["cases"][0]
        
        # Format: Evidence + Question → Answer
        evidence_text = "\n".join([f"{i+1}. {e}" for i, e in enumerate(case["evidence"])])
        
        for question in case["detective_questions"]:
            training_data.append({
                "case_id": case["id"],
                "evidence": evidence_text,
                "question": question,
                "hidden_truth": case["hidden_truth"],
                "false_narrative": case["false_narrative"],
                "answer_keywords": case["solution"]["direct_answer_keywords"],
                "category": case["category"]
            })
    
    return training_data

# Usage:
data = extract_training_data()
print(f"Total training examples: {len(data)}")  # ~1500 (3 questions per case)
```

### Step 2: Split Dataset

```python
from sklearn.model_selection import train_test_split

# 80% train, 10% validation, 10% test
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"Train: {len(train_data)}")      # ~1200
print(f"Validation: {len(val_data)}")   # ~150
print(f"Test: {len(test_data)}")        # ~150
```

---

## 🤖 Model Architecture Options

### Option 1: Fine-tune Existing LLM (RECOMMENDED)

**Best Models for This Task:**
1. **Llama 3.1 8B** - Good reasoning, runs locally
2. **Qwen2.5 7B** - Strong logic capabilities
3. **Mistral 7B** - Balanced performance
4. **Phi-3 Mini** - Lightweight, fast

**Why Fine-tuning?**
- ✅ Already understands language
- ✅ Can learn reasoning patterns from 500 cases
- ✅ Faster training (hours, not days)
- ✅ Better results with less data

**Training Approach:**
```python
# Using Hugging Face Transformers + LoRA
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# Add LoRA adapters (efficient fine-tuning)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Train on Conan cases
# (See detailed code below)
```

---

### Option 2: Train from Scratch (NOT RECOMMENDED)

**Why Not?**
- ❌ Needs 10,000+ cases (you have 500)
- ❌ Takes weeks to train
- ❌ Requires massive compute
- ❌ Worse results than fine-tuning

**Only consider if:**
- You want a tiny specialized model
- You have very limited inference resources

---

## 📝 Training Script (Fine-tuning with LoRA)

```python
# train_conan_model.py
import json
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# 1. Load and prepare data
def load_conan_data():
    """Load all Conan cases and format for training."""
    cases_dir = Path("detective_engine/cases")
    examples = []
    
    for case_file in sorted(cases_dir.glob("C*.json")):
        with open(case_file) as f:
            data = json.load(f)
        
        case = data["cases"][0]
        evidence = "\n".join([f"{i+1}. {e}" for i, e in enumerate(case["evidence"])])
        
        # Create training prompt
        prompt = f"""You are a detective analyzing a case. Given the evidence, identify the hidden truth.

Evidence:
{evidence}

Question: What actually happened? Explain the trick or deception used.

Answer: {case["hidden_truth"]}

False Narrative (REJECT THIS): {case["false_narrative"]}
"""
        
        examples.append({"text": prompt})
    
    return Dataset.from_list(examples)

# 2. Setup model
def setup_model(model_name="meta-llama/Llama-3.1-8B"):
    """Load model with LoRA for efficient fine-tuning."""
    
    # Load model in 4-bit for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

# 3. Training
def train_model():
    """Train the model on Conan cases."""
    
    # Load data
    dataset = load_conan_data()
    train_test = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Setup model
    model, tokenizer = setup_model()
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=2048)
    
    tokenized_train = train_test["train"].map(tokenize_function, batched=True)
    tokenized_test = train_test["test"].map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./conan_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=100,
        warmup_steps=50,
        save_total_limit=3,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
    )
    
    # Train!
    print("🚀 Starting training...")
    trainer.train()
    
    # Save
    model.save_pretrained("./conan_model_final")
    tokenizer.save_pretrained("./conan_model_final")
    
    print("✅ Training complete!")

if __name__ == "__main__":
    train_model()
```

---

## ⚙️ Training Configuration

### Hardware Requirements:

**Minimum (LoRA fine-tuning):**
- GPU: 16GB VRAM (RTX 4080, A4000)
- RAM: 32GB
- Storage: 50GB
- Time: 2-4 hours

**Recommended:**
- GPU: 24GB VRAM (RTX 4090, A5000)
- RAM: 64GB
- Storage: 100GB
- Time: 1-2 hours

**Your Mac M2:**
- ⚠️ Can run inference (testing)
- ❌ Too slow for training (use cloud GPU)

### Cloud Options:
1. **Google Colab Pro** ($10/month) - A100 GPU
2. **RunPod** (~$0.50/hour) - RTX 4090
3. **Lambda Labs** (~$1/hour) - A100

---

## 📈 Training Process

### Phase 1: Setup (30 minutes)
```bash
# Install dependencies
pip install transformers peft datasets accelerate bitsandbytes

# Prepare data
python training_data_extractor.py
```

### Phase 2: Training (2-4 hours)
```bash
# Start training
python train_conan_model.py

# Monitor progress
tensorboard --logdir ./conan_model/runs
```

### Phase 3: Evaluation (30 minutes)
```bash
# Test on held-out cases
python evaluate_model.py
```

---

## 🎯 Success Metrics

### What to Measure:
1. **Accuracy**: Does it identify the correct trick?
2. **Reasoning Quality**: Does it explain the mechanism?
3. **False Narrative Rejection**: Does it avoid the trap?

### Target Performance:
- **Training set**: 90%+ accuracy
- **Validation set**: 75%+ accuracy
- **Test set**: 70%+ accuracy

---

## 🚀 Next Steps

### Immediate (Today):
1. ✅ Create `training_data_extractor.py` script
2. ✅ Extract and inspect training data
3. ✅ Choose base model (Llama 3.1 8B recommended)

### Short-term (This Week):
1. Set up cloud GPU (Google Colab Pro or RunPod)
2. Install dependencies
3. Run first training experiment

### Medium-term (Next 2 Weeks):
1. Fine-tune model on all 500 cases
2. Evaluate on test set
3. Iterate on prompt format if needed

---

## 📚 Resources

### Tutorials:
- [Hugging Face Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Colab Training Notebook](https://colab.research.google.com/drive/1jCkpikz0J2o20FBQmYmAGdiKmJGOMo-o)

### Models to Try:
1. **Llama 3.1 8B** - Best reasoning
2. **Qwen2.5 7B** - Strong logic
3. **Mistral 7B** - Balanced

---

## ✅ Ready to Start?

**You have everything you need:**
- ✅ 500 high-quality training cases
- ✅ Clear training plan
- ✅ Code templates
- ✅ Success metrics

**Next command:**
```bash
# Create the training data extractor
# (I'll create this for you next)
```

Want me to create the training data extraction script now?
