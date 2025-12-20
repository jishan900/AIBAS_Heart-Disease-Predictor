# ActivationBase_cardio_risk
## Subgoal 3 - Docker Data Provision (LearningBase & ActivationBase)

**Docker Installation & Verification**
```bash
docker --version
docker ps
```

**Project Structure Preparation**
```bash
cd ~/Documents/AIBAS_Heart-Disease-Predictor
mkdir -p images/learningBase_cardio_risk
mkdir -p images/activationBase_cardio_risk
```

**Copy Prepared Data Files**
```bash
cp data/Cardiovascular_disease_dataset/training_data.csv images/learningBase_cardio_risk/
cp data/Cardiovascular_disease_dataset/test_data.csv images/learningBase_cardio_risk/
cp data/Cardiovascular_disease_dataset/activation_data.csv images/activationBase_cardio_risk/
```

**Dockerfile - LearningBase**
Path: 
```bash
images/learningBase_cardio_risk/Dockerfile
```
Then, in Dockerfile file: 

```bash
FROM busybox
RUN mkdir -p /tmp/learningBase/train /tmp/learningBase/validation
COPY training_data.csv /tmp/learningBase/train/training_data.csv
COPY test_data.csv     /tmp/learningBase/validation/test_data.csv
COPY README.md         /tmp/learningBase/README.md
```

**Dockerfile - ActivationBase**
Path: 
```bash
images/activationBase_cardio_risk/Dockerfile
```
Then, in Dockerfile file: 

```bash
FROM busybox
RUN mkdir -p /tmp/activationBase
COPY activation_data.csv /tmp/activationBase/activation_data.csv
COPY README.md           /tmp/activationBase/README.md
```

**Build Docker Images (BusyBox-based)**
```bash
docker build -t learningbase_cardio_risk ./images/learningBase_cardio_risk
docker build -t activationbase_cardio_risk ./images/activationBase_cardio_risk
```

**Verify Image Contents**
```bash
docker run --rm learningbase_cardio_risk sh -c "ls -R /tmp/learningBase"
docker run --rm activationbase_cardio_risk sh -c "ls -R /tmp/activationBase"
```

**Tag Images for Docker Hub**
```bash
docker tag learningbase_cardio_risk jishan900/learningbase_cardio_risk:latest
docker tag activationbase_cardio_risk jishan900/activationbase_cardio_risk:latest
```

**Login & Push Images (Public)**
```bash
docker login
docker push jishan900/learningbase_cardio_risk:latest
docker push jishan900/activationbase_cardio_risk:latest 
```

**Docker Pull Commands for Reproducibility**
```bash
docker pull jishan900/learningbase_cardio_risk:latest
docker pull jishan900/activationbase_cardio_risk:latest 
```


## Ownership
Muhammad Farhan Tanvir, Md Asifuzzaman Jishan

## Course and Institution
Created as part of the course “M. Grum: Advanced AI-based Application Systems” by the ‘Junior Chair for Business InformaGon Science, esp. AI-based ApplicaGon Systems’ at University of Potsdam

## Data origin
Scraped/downloaded programmatically from Kaggle: “Cardiovascular Disease Dataset” by sulianova. Dataset: [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset))

## Content / Paths inside the image
- /tmp/activationBase/activation_data.csv
- /tmp/activationBase/Readme.md

## License commitment
AGPL-3.0 license commitment.
