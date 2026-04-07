# 🕵️ Detective Conan Reasoning Engine

A research-grade reasoning evaluation system with 500+ training cases for neural network training.

## 📊 Project Status

- ✅ **500 detective cases** generated (C001-C500)
- ✅ **1,500 training examples** extracted (3 per case)
- ✅ **15 Detective Conan trick templates** implemented
- ✅ **100% validation pass rate**
- ✅ **Ready for neural network training**

---

## 📁 Project Structure

```
Conan/
├── detective_engine/          # Core reasoning engine
│   ├── cases/                 # 500 case files (C001-C500)
│   ├── perception.py          # Perception integrity framework
│   ├── causality.py           # Causal reasoning engine
│   └── validator.py           # Case validation system
│
├── training_data/             # ML-ready training data
│   ├── train.json             # 1,200 training examples (80%)
│   ├── val.json               # 150 validation examples (10%)
│   ├── test.json              # 150 test examples (10%)
│   ├── all_examples.json      # All 1,500 examples
│   └── statistics.txt         # Dataset statistics
│
├── scripts/
│   ├── generators/            # Active generation tools
│   │   ├── batch_generator.py         # Automated batch generation
│   │   ├── generate_all.py            # Master generation script
│   │   ├── extract_training_data.py   # Training data extraction
│   │   └── conan_template_generator.py # Interactive generator
│   │
│   └── legacy/                # Experimental/legacy tools
│       ├── controlled_case_pipeline.py
│       ├── human_guided_pipeline.py
│       ├── quick_case_builder.py
│       └── template_case_generator.py
│
├── templates/                 # Case generation templates
│   ├── templates_part1.py     # Physical deception (5 templates)
│   ├── templates_part2.py     # Audio/visual & timing (5 templates)
│   └── templates_part3.py     # Environmental & psychological (5 templates)
│
├── tests/                     # Test files
│   ├── test_perception.py
│   └── test_causality.py
│
├── docs/                      # Documentation
│   ├── PROJECT_COMPLETE.md    # Project completion summary
│   ├── TRAINING_PLAN.md       # Neural network training guide
│   ├── GENERATION_PLAN.md     # Case generation strategy
│   ├── TEMPLATE_GUIDE.md      # Template usage guide
│   ├── QUALITY_COMPARISON.md  # Quality analysis
│   ├── METADATA_ANALYSIS.md   # Evidence metadata analysis
│   ├── DELIVERY_SUMMARY.md    # Delivery summary
│   └── README_GENERATION.md   # Generation quick start
│
├── main.py                    # Interactive case testing
└── README.md                  # This file
```

---

## 🚀 Quick Start

### 1. Test Existing Cases
```bash
# Interactive testing
python3 main.py

# Validate all cases
python3 main.py --validate
```

### 2. Generate More Cases
```bash
# Generate 100 more cases
cd scripts/generators
python3 batch_generator.py --count 100 --start C501

# Or use master script
python3 generate_all.py
```

### 3. Extract Training Data
```bash
cd scripts/generators
python3 extract_training_data.py
```

### 4. Start Training
See `docs/TRAINING_PLAN.md` for complete neural network training guide.

---

## 📚 Documentation

### Getting Started
- **README.md** (this file) - Project overview
- **docs/PROJECT_COMPLETE.md** - Completion summary
- **docs/README_GENERATION.md** - Quick generation guide

### Case Generation
- **docs/GENERATION_PLAN.md** - Generation strategy
- **docs/TEMPLATE_GUIDE.md** - Template usage
- **docs/QUALITY_COMPARISON.md** - Quality analysis

### Neural Network Training
- **docs/TRAINING_PLAN.md** - Complete training guide
- **docs/METADATA_ANALYSIS.md** - Evidence metadata analysis

---

## 🎯 Use Cases

### 1. Interactive Case Testing
```bash
python3 main.py
# Enter case ID: C100
```

### 2. Batch Case Generation
```bash
cd scripts/generators
python3 batch_generator.py --count 50 --template ice_weapon
```

### 3. Training Data Extraction
```bash
cd scripts/generators
python3 extract_training_data.py
# Output: training_data/ folder with train/val/test splits
```

### 4. Neural Network Training
```python
# See docs/TRAINING_PLAN.md for complete code
from transformers import AutoModelForCausalLM
import json

# Load training data
with open("training_data/train.json") as f:
    train_data = json.load(f)

# Fine-tune model (LoRA)
# ... (see training plan)
```

---

## 📊 Dataset Statistics

### Cases
- **Total**: 500 cases (C001-C500)
- **Original**: 6 hand-crafted cases
- **Generated**: 494 template-based cases
- **Templates**: 15 Detective Conan tricks

### Training Data
- **Total examples**: 1,500
- **Train**: 1,200 (80%)
- **Validation**: 150 (10%)
- **Test**: 150 (10%)

### Categories
- Physics deception: 300 examples
- Chemistry deception: 291 examples
- Staging deception: 195 examples
- Mechanical deception: 102 examples
- Other: 612 examples

---

## 🎭 Available Templates

### Physical Deception (templates_part1.py)
1. **Tape Locked Room** - Door taped from inside trick
2. **Ice Weapon** - Melting ice dagger
3. **Fishing Line** - Remote trigger mechanism
4. **Poison Ice Cubes** - Delayed poison delivery
5. **Mirror Reflection** - Optical illusion alibi

### Audio/Visual & Timing (templates_part2.py)
6. **Voice Recording** - Pre-recorded audio
7. **Elevator Timing** - Stopped elevator trick
8. **Blood Transfer** - False crime scene
9. **Double Identity** - Twin/disguise
10. **Staged Suicide** - Murder as suicide

### Environmental & Psychological (templates_part3.py)
11. **Air Pressure Door** - Temporary lock
12. **Poison Contact Lens** - Dissolving delivery
13. **Reverse Locked Room** - Body moved in
14. **Heat-Activated Poison** - Temperature trigger
15. **Psychological Trigger** - Manipulation

---

## 🔧 Development

### Generate New Template
```python
# Edit templates/templates_part1.py
TEMPLATES_PART1["new_trick"] = {
    "name": "New Trick Name",
    "description": "One-line description",
    "category": "category-name",
    "insight": {...},
    "evidence_template": [...],
    "hidden_truth": "...",
    "false_narrative": "...",
    "keywords": {...}
}
```

### Validate Cases
```bash
python3 main.py --validate
```

### Run Tests
```bash
cd tests
python3 test_perception.py
python3 test_causality.py
```

---

## 📈 Training Recommendations

### Recommended Models
1. **Llama 3.1 8B** - Best reasoning capabilities
2. **Qwen2.5 7B** - Strong logic and reasoning
3. **Mistral 7B** - Balanced performance

### Training Method
- **LoRA fine-tuning** (recommended)
- Training time: 2-4 hours on cloud GPU
- Expected accuracy: 70-90%

### Hardware
- **Cloud GPU**: Google Colab Pro ($10/month), RunPod ($0.50/hour)
- **Local GPU**: RTX 4080/4090 (16-24GB VRAM)

See `docs/TRAINING_PLAN.md` for complete guide.

---

## 🎓 Research Applications

### Reasoning Evaluation
- Test LLM reasoning capabilities
- Benchmark deception detection
- Evaluate evidence weighting

### Training Applications
- Fine-tune reasoning models
- Train evidence ranking systems
- Develop bias detection models

### Research Areas
- Causal reasoning
- Perception integrity
- Cognitive bias detection
- Structured reasoning evaluation

---

## 📝 Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{conan_reasoning_2024,
  title={Detective Conan Reasoning Engine: 500 Cases for Structured Reasoning Evaluation},
  author={Your Name},
  year={2024},
  description={500 detective cases with 15 deception patterns for neural network training}
}
```

---

## 🤝 Contributing

### Add New Templates
1. Create template in `templates/templates_part*.py`
2. Test with `scripts/generators/conan_template_generator.py`
3. Generate batch with `scripts/generators/batch_generator.py`

### Improve Quality
1. Add evidence_meta to cases (see `docs/METADATA_ANALYSIS.md`)
2. Enhance detective questions
3. Add more contradiction types

---

## 📞 Support

### Documentation
- `docs/` folder contains all guides
- `docs/TRAINING_PLAN.md` for training help
- `docs/TEMPLATE_GUIDE.md` for generation help

### Issues
- Check validation: `python3 main.py --validate`
- Review statistics: `training_data/statistics.txt`
- Test interactively: `python3 main.py`

---

## ✅ Project Milestones

- [x] Core reasoning engine implemented
- [x] 15 Detective Conan templates created
- [x] 500 cases generated
- [x] Training data extracted (1,500 examples)
- [x] Quality validation (100% pass rate)
- [x] Documentation complete
- [ ] Neural network training (next step)
- [ ] Model evaluation
- [ ] Research paper publication

---

## 🎉 Acknowledgments

Inspired by Detective Conan (Case Closed) anime/manga series by Gosho Aoyama.

---

**Status**: ✅ Ready for neural network training

**Last Updated**: 2024

**Version**: 1.0.0
