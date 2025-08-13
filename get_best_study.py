import optuna
storage_name = "sqlite:///data/my_study.db"
study_name = "encoder-optimization-v1"  # Give your study a name
loaded_study = optuna.load_study(study_name=study_name, storage=storage_name)
print(loaded_study.best_params)


