from pathlib import Path
import subprocess
import pendulum
import os
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

# =========================================================
# Docker-safe project root
# =========================================================
PROJECT_ROOT = Path("/opt/project")
PYTHON = "python"  # container Python

# =========================================================
# Helper to run training scripts with logging
# =========================================================
def run_script(relative_path):
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["WANDB_SILENT"] = "true"
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)

    script_path = PROJECT_ROOT / relative_path

    print(f"Running script: {script_path}")
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    result = subprocess.run(
        [PYTHON, str(script_path)],
        capture_output=True,
        text=True,
        env=env,
    )
    print("stdout:\n", result.stdout)
    print("stderr:\n", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Script {relative_path} failed with exit code {result.returncode}")

# =========================================================
# DAG defaults
# =========================================================
default_args = {
    "owner": "carbon_trading",
    "retries": 1,
    "retry_delay": pendulum.duration(minutes=5),
}

# =========================================================
# DAG definition
# =========================================================
with DAG(
    dag_id="carbon_trading_pipeline",
    description="ML pipeline for carbon trading forecasting",
    default_args=default_args,
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule="@daily",
    catchup=False,
    tags=["carbon_trading", "ml"],
) as dag:

    # -----------------------------------------------------
    # Step 1: Check data availability
    # -----------------------------------------------------
    check_data = BashOperator(
        task_id="check_data",
        bash_command="""
        if [ -z "$(ls /opt/project/data 2>/dev/null)" ]; then
            echo "❌ data/ directory empty or missing"
            exit 1
        else
            echo "✅ data present"
        fi
        """
    )

    # -----------------------------------------------------
    # Step 2: Feature engineering
    # -----------------------------------------------------
    feature_engineering = PythonOperator(
        task_id="feature_engineering",
        python_callable=run_script,
        op_args=["training/feature_engineering.py"],
    )

    # -----------------------------------------------------
    # Step 3: Train models
    # -----------------------------------------------------
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

        # Sequential execution
        train_arima >> train_prophet >> train_rf

    # -----------------------------------------------------
    # Step 4: Archive outputs
    # -----------------------------------------------------
    archive_outputs = BashOperator(
        task_id="archive_outputs",
        bash_command="""
        DATE=$(date +%Y%m%d)
        mkdir -p /opt/project/outputs/$DATE
        cp -r /opt/project/models/* /opt/project/outputs/$DATE/ || true
        echo "✅ Models archived to outputs/$DATE"
        """
    )

    # -----------------------------------------------------
    # Step 5: Pipeline done
    # -----------------------------------------------------
    pipeline_done = BashOperator(
        task_id="pipeline_done",
        bash_command="echo '✅ Carbon trading pipeline completed successfully'",
    )

    # -----------------------------------------------------
    # DAG flow
    # -----------------------------------------------------
    check_data >> feature_engineering >> train_models >> archive_outputs >> pipeline_done
