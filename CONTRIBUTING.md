# Contributing to Detective Conan Reasoning Engine

Thank you for your interest in contributing! This project welcomes contributions from the community.

## 🎯 Ways to Contribute

### 1. Add New Case Templates
- Create new Detective Conan trick templates
- Add to `templates/` folder
- Follow existing template structure
- Test with batch generator

### 2. Improve Case Quality
- Add evidence_meta to existing cases
- Enhance detective questions
- Add more contradiction types
- Improve hidden truth explanations

### 3. Enhance Documentation
- Fix typos and errors
- Add examples and tutorials
- Translate documentation
- Create video tutorials

### 4. Report Bugs
- Use GitHub Issues
- Provide reproduction steps
- Include error messages
- Suggest fixes if possible

### 5. Improve Code
- Optimize generation scripts
- Add new features
- Improve validation
- Add tests

## 📝 Contribution Process

### 1. Fork the Repository
```bash
git clone https://github.com/yourusername/conan-reasoning-engine.git
cd conan-reasoning-engine
```

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes
- Follow existing code style
- Add docstrings
- Update documentation
- Add tests if applicable

### 4. Test Your Changes
```bash
# Validate cases
python3 main.py --validate

# Test generation
cd scripts/generators
python3 batch_generator.py --count 1 --start C999
```

### 5. Commit and Push
```bash
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
```

### 6. Create Pull Request
- Describe your changes
- Reference related issues
- Wait for review

## 🎨 Code Style

### Python
- Follow PEP 8
- Use type hints where possible
- Add docstrings to functions
- Keep functions focused and small

### Documentation
- Use Markdown
- Include code examples
- Add screenshots if helpful
- Keep it concise

## 🧪 Testing

### Before Submitting
- [ ] All cases validate successfully
- [ ] Generation scripts work
- [ ] Documentation is updated
- [ ] No broken links
- [ ] Code is formatted

### Run Tests
```bash
cd tests
python3 test_perception.py
python3 test_causality.py
```

## 📋 Template Contribution Guidelines

### New Template Structure
```python
"template_key": {
    "name": "Template Name",
    "description": "One-line description",
    "category": "category-name",
    "insight": {
        "title": "...",
        "reasoning_type": "...",
        "summary": "...",
        "transfer_rule": "..."
    },
    "evidence_template": [
        # 6 evidence items with {variables}
    ],
    "hidden_truth": "...",
    "false_narrative": "...",
    "keywords": {
        "answer": [...],
        "fact_a": [...],
        "fact_b": [...]
    }
}
```

### Template Requirements
- Must have exactly 6 evidence items
- Must include {victim}, {suspect}, {location} variables
- Must have clear hidden truth and false narrative
- Must include relevant keywords

## 🐛 Bug Reports

### Include:
- Python version
- Operating system
- Error message (full traceback)
- Steps to reproduce
- Expected vs actual behavior

### Example
```markdown
**Bug**: Generation fails with template X

**Environment**:
- Python 3.9
- macOS 13.0

**Steps**:
1. Run `python3 batch_generator.py --template X`
2. Error occurs

**Error**:
```
[paste error message]
```

**Expected**: Should generate case successfully
```

## 💡 Feature Requests

### Include:
- Clear description
- Use case
- Example if possible
- Why it's valuable

## 📞 Questions?

- Open a GitHub Issue
- Tag with "question"
- We'll respond within 48 hours

## 🏆 Recognition

Contributors will be:
- Listed in README.md
- Mentioned in release notes
- Credited in research papers (if applicable)

## 📜 Code of Conduct

### Our Standards
- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the project

### Unacceptable Behavior
- Harassment or discrimination
- Trolling or insulting comments
- Personal attacks
- Publishing private information

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🎉
