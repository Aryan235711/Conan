
# 🕵️ Detective Conan Reasoning Engine

A research-grade reasoning evaluation system for detective-style reasoning tasks. This public repository contains the core engine, test harness, and documentation for research and educational use.

## 📁 Project Structure (Public Release)

```
Conan/
├── detective_engine/          # Core reasoning engine (public modules only)
│   ├── perception.py          # Perception integrity framework
│   ├── causality.py           # Causal reasoning engine
│   └── validator.py           # Case validation system
│
├── tests/                     # Test files
│   ├── test_perception.py
│   └── test_causality.py
│
├── docs/                      # Documentation (public)
│   ├── PROJECT_COMPLETE.md
│   ├── TRAINING_PLAN.md
│   ├── GENERATION_PLAN.md
│   ├── TEMPLATE_GUIDE.md
│   ├── QUALITY_COMPARISON.md
│   ├── METADATA_ANALYSIS.md
│   ├── DELIVERY_SUMMARY.md
│   └── README_GENERATION.md
│
├── main.py                    # Interactive case testing
└── README.md                  # This file
```

> **Note:**
> - Some files and folders referenced in earlier versions (such as 500+ detective cases, automation scripts, and full training datasets) are not included in this public repository. These were internal or generated assets and are intentionally omitted for privacy, licensing, or size reasons.
> - The `.gitignore` excludes local development files, virtual environments, Jupyter notebooks, and other non-essential or large files. Only the core engine, tests, and documentation are included here.

---



## 🚀 Quick Start

### 1. Run the Engine

```bash
python3 main.py
```

### 2. Run Tests

```bash
python3 -m unittest discover tests
```

---


## 📚 Documentation

- **README.md** (this file) - Project overview
- **docs/PROJECT_COMPLETE.md** - Completion summary
- **docs/README_GENERATION.md** - Quick generation guide
- **docs/GENERATION_PLAN.md** - Generation strategy
- **docs/TEMPLATE_GUIDE.md** - Template usage
- **docs/QUALITY_COMPARISON.md** - Quality analysis
- **docs/TRAINING_PLAN.md** - Training guide (for reference only)
- **docs/METADATA_ANALYSIS.md** - Evidence metadata analysis

---

## 🎭 Templates (Public)

The public repository includes only the template framework and documentation. No actual case data, generated cases, or training data are included.

See `docs/TEMPLATE_GUIDE.md` for template structure and usage.

---

## 🤝 Contributing

See `CONTRIBUTING.md` for contribution guidelines.

---

## 📞 Support

For help, see the `docs/` folder or open a GitHub issue.

---

**Status**: Public engine and template framework only. No case data or training data included.

**Last Updated**: 2024

**Version**: 1.0.0
