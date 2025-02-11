from src import CodeValidator

# Create validator in test mode
validator = CodeValidator(test_mode=True)

# Test a simple completion
result = validator.generate_completion("def add(a, b):\n    return")
print(f"Generated completion: {result}")

# Test metrics
metrics = validator.evaluate_quality(result, "a + b")
print(f"Quality metrics: {metrics}") 