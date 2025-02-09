SECURITY_PATTERNS = [
    'os.system', 'subprocess.run',
    'eval(', 'exec(', 'pickle.load',
    'shutil.rmtree', '__import__'
]

METRIC_THRESHOLDS = {
    'bleu': 0.65,
    'rouge': 0.75,
    'exact_match': 1.0,
    'syntax_valid': True,
    'safety_pass': True
}