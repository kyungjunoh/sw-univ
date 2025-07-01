from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument("--train_file", type=str, default="src/data/data/train.csv", help="Path to the training data file")
    parser.add_argument("--train_add_file", type=str, default="src/data/data/train_add.csv", help="Path to the validation data file")
    parser.add_argument("--test_file", type=str, default="src/data/data/test.csv", help="Path to the test data file")
    parser.add_argument("--submission_file", type=str, default="src/data/data/sample_submission.csv", help="Path to the submission file")
    parser.add_argument("--base_file", type=str, default="src/data/data/base.csv", help="Path to the base data file")
    parser.add_argument("--model_name", type=str, default="xgboost", help="Name of the model to use")
    parser.add_argument("--custom_objective", type=str, default="src/utils/loss.py", help="Path to the custom objective function")
    parser.add_argument("--custom_score", type=str, default="src/utils/compute_metric.py", help="Path to the custom score function")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for hyperparameter tuning")
    return parser.parse_args()