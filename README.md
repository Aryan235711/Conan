
# Detective Conan Reasoning Engine

![MIT License](https://img.shields.io/badge/license-MIT-green.svg)
[![tests](https://github.com/Aryan235711/Conan/actions/workflows/tests.yml/badge.svg)](https://github.com/Aryan235711/Conan/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-orange)

A research-grade reasoning evaluation engine for detective-style logic and AI tasks. This public repository contains the core engine, test harness, and documentation for research and educational use.

**Quick Links:**
- [Getting Started](#-quick-start)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---



## Repository Structure

```
├── .gitignore
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── detective_engine/
│   ├── __init__.py
│   ├── cases/
│   │   ├── C001_silent_room.json
│   │   ├── C002_fingerprint_paradox.json
│   │   ├── C003_quiet_departure.json
│   │   ├── C004_broken_timeline.json
│   │   ├── C005_grieving_partner.json
│   │   └── C006_poisoned_philanthropist.json
│   └── engine/
│       ├── __init__.py
│       ├── bayesian_validator.py
│       ├── case_loader.py
│       ├── case_runner.py
│       ├── case_validator.py
│       ├── causality_validator.py
│       ├── insight_graph.py
│       ├── llm_judge.py
│       ├── models.py
│       ├── perception_integrity.py
│       ├── phase_generator.py
│       ├── reasoning_graph.py
│       ├── reasoning_parser.py
│       ├── reward_interface.py
│       ├── user_profile.py
│       └── validator.py
├── docs/
│   ├── PROJECT_COMPLETE.md
│   └── QUALITY_COMPARISON.md
├── main.py
└── tests/
    ├── run_all.py
    ├── test_causality.py
    ├── test_integration.py
    ├── test_llm_judge.py
    ├── test_perception.py
    ├── test_reasoning_parser.py
    └── test_reward_interface.py
```


> **Note:**
> - This public repo includes only the core engine, tests, and documentation. No private/internal datasets, automation scripts, or large generated assets are included.
> - The `.gitignore` excludes local development files, virtual environments, Jupyter notebooks, and other non-essential or large files.

---




## Quick Start


### 1. Run the Engine

```bash
python3 main.py
```

### 2. Run Tests

```bash
python3 tests/run_all.py
```

The test files are standalone scripts, so `unittest discover` does not find them. The runner executes each one and exits non-zero if any fail.

---





## Documentation

- **README.md** (this file) - Project overview
- **CONTRIBUTING.md** - Contribution guidelines
- **docs/PROJECT_COMPLETE.md** - Completion summary
- **docs/QUALITY_COMPARISON.md** - Quality analysis


> Note: Only files listed above are present in the public repository. No large datasets, automation scripts, or private/internal data are included.

---


## Templates (Public)

This repository includes only the template framework and documentation. No actual case data, generated cases, or training data are included.

See `docs/TEMPLATE_GUIDE.md` for template structure and usage.

---


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

---


## Support

For help, see the `docs/` folder or open a GitHub issue.

---


---

**Status:** Public engine and template framework only. No case data or training data included.

**Last Updated:** 2026

**Version:** 1.0.0
