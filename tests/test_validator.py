import warnings
import pytest
from src.validator import CodeValidator
from src.config import METRIC_THRESHOLDS
import json
import os

# Filter out specific deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google._upb._message")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils._pytree")

@pytest.fixture(scope="module")
def validator():
    return CodeValidator(test_mode=True)

def load_test_cases():
    test_dir = os.path.join(os.path.dirname(__file__), 'test_cases')
    with open(os    .path.join(test_dir, 'sample_cases.json')) as f:
        return json.load(f)['test_cases']

@pytest.mark.parametrize("case", load_test_cases())
def test_code_completion(validator, case):
    # Test full completion
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

@pytest.mark.parametrize("case", load_test_cases())
def test_next_token_prediction(validator, case):
    # Test single token prediction
    next_token = validator.generate_next_token(case['context'])
    assert next_token.strip() in case['expected'], \
        f"Predicted token '{next_token}' not in expected completion '{case['expected']}'"

@pytest.mark.parametrize("case", load_test_cases())
def test_token_probabilities(validator, case):
    # Test token probability distribution
    token_probs = validator.get_next_token_probabilities(case['context'], top_k=5)
    
    # Check structure
    assert len(token_probs) > 0, "No token probabilities returned"
    assert all(isinstance(t, str) and isinstance(p, float) for t, p in token_probs), \
        "Invalid probability distribution format"
    
    # Check probability values
    assert all(0 <= p <= 1 for _, p in token_probs), \
        "Probabilities should be between 0 and 1"
    
    # In test mode, should return exactly one token with probability 1
    if validator.test_mode:
        assert len(token_probs) == 1, "Test mode should return single token"
        assert token_probs[0][1] == 1.0, "Test mode should return probability of 1.0"

def test_temperature_sampling(validator):
    """Test different temperature settings for completion generation"""
    context = "def add(a, b):\n    return"
    
    # Test different temperatures
    for temp in [0.0, 0.7, 1.0]:
        completion = validator.generate_completion(context, temperature=temp)
        assert completion.strip() == "a + b", \
            f"Temperature {temp} produced unexpected completion: {completion}" 