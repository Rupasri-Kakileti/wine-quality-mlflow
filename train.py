import os
import warnings
import sys
import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Check if the wine-quality.csv file exists
    wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wine-quality.csv")
    if not os.path.exists(wine_path):
        print(f"Error: {wine_path} does not exist!")
        sys.exit(1)

    # Read the wine-quality csv file
    data = pd.read_csv(wine_path)

    # Split the data into training and test sets (0.75, 0.25)
    train, test = train_test_split(data)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Model parameters
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run() as run:
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # Log parameters and metrics to MLflow
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Register the model in MLflow model registry with versioning
        model_name = "ElasticNetWineModel"
        model_uri = f"runs:/{run.info.run_id}/model"
        registered_model = mlflow.register_model(model_uri, model_name)

        # Tag the model version
        latest_version = registered_model.version
        mlflow.log_param("model_version", latest_version)
        print(f"Model registered as version {latest_version}.")

    # Automatically push changes to GitLab and tag with version
    try:
        version_tag = f"v{latest_version}"
        print(f"Preparing to push changes to GitLab with tag {version_tag}.")

        # Add changes to Git
        subprocess.run(["git", "add", "."], check=True)

        # Commit changes
        commit_message = f"Logged metrics to MLflow, updated code, version {version_tag}"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)

        # Tag the commit
        subprocess.run(["git", "tag", version_tag], check=True)

        # Push changes and the tag
        subprocess.run(["git", "push", "origin", "main"], check=True)  # Adjust branch if necessary
        subprocess.run(["git", "push", "origin", version_tag], check=True)

        print(f"Changes pushed to GitLab with tag {version_tag}.")
    except subprocess.CalledProcessError as e:
        print("Error pushing to GitLab:", e)
