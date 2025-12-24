from pathlib import Path
import subprocess
import pendulum
import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

# detect project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PYTHON = (
    PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    if (PROJECT_ROOT / "venv" / "Scripts").exists()
    else PROJECT_ROOT / "venv" / "bin" / "python"
)

def run_script(relative_path):
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["WANDB_SILENT"] = "true"
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)
    script_path = PROJECT_ROOT / relative_path
    subprocess.run([str(PYTHON), str(script_path)], check=True, env=env,)

default_args = {
    "owner": "carbon_trading",
    "retries": 1,
    "retry_delay": pendulum.duration(minutes=5),
}

# dag definition
with DAG(
    dag_id="carbon_trading_pipeline",
    description="ML pipeline for carbon trading data forecasting",
    default_args=default_args,
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule="@daily",
    catchup=False,
    tags=["carbon_trading", "ml"],
) as dag:

    # step 1: check data availability
    check_data = BashOperator(
        task_id="check_data",
        bash_command=f"""
        if [ -z "$(ls {PROJECT_ROOT}/data)" ]; then
            echo "❌ data/ directory empty or missing"
            exit 1
        else
            echo "✅ data present"
        fi
        """
    )

    # step 2: feature engineering
    feature_engineering = PythonOperator(
        task_id="feature_engineering",
        python_callable=run_script,
        op_args=["training/feature_engineering.py"],
    )

    # step 3: train models
    with TaskGroup("train_models") as train_models:

        train_arima = PythonOperator(
            task_id="train_arima",
            python_callable=run_script,
            op_args=["training/train_arima.py"],
        )

        train_prophet = PythonOperator(
            task_id="train_prophet",
            python_callable=run_script,
            op_args=["training/train_prophet.py"],
        )

        train_rf = PythonOperator(
            task_id="train_random_forest",
            python_callable=run_script,
            op_args=["training/train_rf.py"],
        )

        [train_arima, train_prophet, train_rf]

    # step 4: archive outputs
    archive_outputs = BashOperator(
        task_id="archive_outputs",
        bash_command=f"""
        mkdir -p {PROJECT_ROOT}/outputs/$(date +%Y%m%d)
        cp -r {PROJECT_ROOT}/models/* {PROJECT_ROOT}/outputs/$(date +%Y%m%d)/ || true
        """
    )

    # final step: pipeline done
    pipeline_done = BashOperator(
        task_id="pipeline_done",
        bash_command="echo '✅ Carbon trading pipeline completed successfully'",
    )

    # define dag flow
    check_data >> feature_engineering >> train_models >> archive_outputs >> pipeline_done