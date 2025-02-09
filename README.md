# NTP-AT (Next Token Prediction - Automated Testing)

A Python-based validation system for testing and evaluating Next Token Prediction models in code completion tasks. This system provides automated testing capabilities to ensure the quality, accuracy, and safety of next-token predictions in programming contexts.

## Features

- Next token prediction validation for code completions
- Multiple evaluation metrics:
  - BLEU score for prediction quality
  - ROUGE score for sequence alignment
  - Exact Match for perfect predictions
- Python syntax validation
- Security pattern scanning
- Configurable quality thresholds
- Comprehensive test suite
- CI/CD pipeline with GitHub Actions

## Installation

```bash
# Clone the repository
git clone https://github.com/oCt-raiN/NTP-AT.git
cd NTP-AT

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from src.validator import CodeValidator

# Initialize the validator
validator = CodeValidator()

# Test next token prediction
context = "def add(a, b):\n    return"  # Incomplete code context
completion = validator.generate_completion(context)  # Get next token prediction
metrics = validator.evaluate_quality(completion, "a + b")  # Evaluate prediction

print(f"Predicted next tokens: {completion}")
print(f"Quality metrics: {metrics}")
```

## Testing

Run the test suite:
```bash
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

## Project Structure

```
code-validator/
├── src/
│   ├── validator.py    # Core NTP validation engine
│   └── config.py       # Configuration settings
├── tests/
│   ├── test_validator.py
│   └── test_cases/
│       └── sample_cases.json
├── .github/
│   └── workflows/
│       └── validate.yml
├── pyproject.toml
└── requirements.txt
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 