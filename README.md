create a virtual environment in linux, or wsl (if using windows) and install all the necessary packages (requirements.txt)


then run 
cd training
python train_arima.py
python train_prophet.py
python train_rf.py

then it will save the .pkl model file in models/

create your api key in comet website and put it in comet-config
create your api key in wandb and put in terminal when running for first time

now run app.py

For airflow, run airflow standalone

it will display username and password when running it for the first time, and will be saved in a json file for future runs. only for the first run, these will have to be entered in the ui.

go to carbon_trading_pipeline in the list of dags and trigger it. this will cause the entire pipeline to be run once.


pull the docker image apache airflow
