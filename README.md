# ML Project — CI/CD Pipeline

## Files
- `model.py` — trains a Decision Tree on Iris dataset
- `test_model.py` — 3 tests for the model
- `requirements.txt` — dependencies

## How CI/CD Works
Every push to main automatically:
1. Installs scikit-learn
2. Runs the model
3. Runs all tests
