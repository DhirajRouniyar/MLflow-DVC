# MLflow-DVC

...
## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional] #Which you dont want to share with the user
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml
10. app.py

...
# How to run?
...
### STEPS:

clone the repository

```bash
https://github.com/DhirajRouniyar/MLflow-DVC
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n cnncls python=3.11 -y
```

```bash
conda activate cnncls
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

#### cmd
- mlflow ui

#### dagshub
[dagshub](https://dagshub.com/)
```bash
import dagshub
dagshub.init(repo_owner='DhirajRouniyar', repo_name='MLflow-DVC', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

```
