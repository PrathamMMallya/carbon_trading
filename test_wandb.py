import wandb
wandb.init(project="carbon-price-forecasting", name="test_run")
wandb.log({"acc": 0.9})
