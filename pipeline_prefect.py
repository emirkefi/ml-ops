import subprocess
import sys
import argparse

from prefect import task, flow

from model_pipeline import (
    prepare_data,
    train_model,
    save_model,
    load_model,
    evaluate_model,
)

# TASK 1 : INSTALL DEPENDENCIES


@task
def install_dependencies():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True
    )
    print("Dependencies installed successfully")


# TASK 2 : PREPARE DATA


@task
def prepare_data_task():
    data = prepare_data()
    print("Data prepared successfully")
    return data


# TASK 3 : TRAIN MODEL


@task
def train_model_task(data):
    X_train, X_test, y_train, y_test = data

    model = train_model(X_train, y_train)

    print("Model trained successfully")

    return model, X_test, y_test


# ==========================================
# TASK 4 : SAVE MODEL
# ==========================================


@task
def save_model_task(model):
    save_model(model)
    print("Model saved successfully")


# ==========================================
# TASK 5 : LOAD MODEL
# ==========================================


@task
def load_model_task():
    model = load_model()
    print("Model loaded successfully")
    return model


# ==========================================
# TASK 6 : EVALUATE MODEL
# ==========================================


@task
def evaluate_model_task(model, X_test, y_test):
    result = evaluate_model(model, X_test, y_test)
    print("Model evaluated successfully")
    return result


# ==========================================
# TASK 7 : CODE FORMATTING
# ==========================================


@task
def format_code():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "black",
            "model_pipeline.py",
            "main.py",
            "pipeline_prefect.py",
        ],
        check=True,
    )


# ==========================================
# TASK 8 : CODE QUALITY
# ==========================================


@task
def check_code_quality():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pylint",
            "model_pipeline.py",
            "main.py",
            "pipeline_prefect.py",
        ],
        check=False,
    )


# ==========================================
# TASK 9 : CODE SECURITY
# ==========================================


@task
def check_security():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "bandit",
            "-ll",
            "model_pipeline.py",
            "main.py",
            "pipeline_prefect.py",
        ],
        check=False,
    )


# ==========================================
# TASK 10 : UNIT TESTS
# ==========================================


@task
def run_tests():
    subprocess.run([sys.executable, "-m", "pytest", "-v"], check=True)


# ==========================================
# FLOW CODE
# ==========================================


@flow(name="code")
def code_flow():
    format_code()
    check_code_quality()
    check_security()
    run_tests()


# ==========================================
# FLOW ALL
# ==========================================


@flow(name="all")
def all_flow():

    install_dependencies()

    code_flow()

    data = prepare_data_task()

    model, X_test, y_test = train_model_task(data)

    save_model_task(model)

    evaluate_model_task(model, X_test, y_test)


# ==========================================
# FLOW TRAIN
# ==========================================


@flow(name="train")
def train_flow():

    install_dependencies()

    data = prepare_data_task()

    model, X_test, y_test = train_model_task(data)

    save_model_task(model)


# ==========================================
# FLOW EVALUATE
# ==========================================


@flow(name="evaluate")
def evaluate_flow():

    model = load_model_task()

    data = prepare_data_task()

    X_train, X_test, y_train, y_test = data

    evaluate_model_task(model, X_test, y_test)


# ==========================================
# MAIN : EXECUTION
# ==========================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--flow",
        choices=["all", "train", "entrainement", "evaluate", "code"],
        default="all",
    )

    args = parser.parse_args()

    if args.flow == "all":
        all_flow()

    elif args.flow in ["train", "entrainement"]:
        train_flow()

    elif args.flow == "evaluate":
        evaluate_flow()

    elif args.flow == "code":
        code_flow()
