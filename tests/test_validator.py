import pytest
from src.validator import CodeValidator
from src.config import METRIC_THRESHOLDS
import json
import os

@pytest.fixture(scope="module")
def validator():
    return CodeValidator(test_mode=True)  # Enable test mode

def load_test_cases():
    test_dir = os.path.join(os.path.dirname(__file__), 'test_cases')
    with open(os.path.join(test_dir, 'sample_cases.json')) as f:
        return json.load(f)['test_cases']

@pytest.mark.parametrize("case", load_test_cases())
def test_code_completion(validator, case):
    generated = validator.generate_completion(case['context'])
    metrics = validator.evaluate_quality(generated, case['expected'])
    
    assert metrics['syntax_valid'] == METRIC_THRESHOLDS['syntax_valid'], \
        f"Syntax error in:\n{generated}"
        
    assert metrics['safety_pass'] == METRIC_THRESHOLDS['safety_pass'], \
        f"Security violation in:\n{generated}"
        
    assert metrics['bleu'] >= METRIC_THRESHOLDS['bleu'], \
        f"BLEU score {metrics['bleu']:.2f} < {METRIC_THRESHOLDS['bleu']}"

    assert metrics['rouge'] >= METRIC_THRESHOLDS['rouge'], \
        f"ROUGE-L score {metrics['rouge']:.2f} < {METRIC_THRESHOLDS['rouge']}" 