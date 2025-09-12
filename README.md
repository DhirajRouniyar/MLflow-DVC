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
```bash
dvc init
dvc repro
dvc dag
```
### AWS-CICD-Deployment-with-Github-Actions
# 1. Login to AWS console.
# 2. Create IAM user for deployment
```bash
#with specific access

1. EC2 access : It is virtual machine
2. ECR: Elastic Container registry to save your docker image in aws

#Description: About the deployment

1. Build docker image of the source code

2. Push your docker image to ECR

3. Launch Your EC2 

4. Pull Your image from ECR in EC2

5. Lauch your docker image in EC2

#Policy:

1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess
```
# 3. Create ECR repo to store/save docker image
- Save the URI: 290690312692.dkr.ecr.us-east-2.amazonaws.com/mlflow
# 4. Create EC2 machine (Ubuntu)
