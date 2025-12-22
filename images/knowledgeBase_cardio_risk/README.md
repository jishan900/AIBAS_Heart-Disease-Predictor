# KnowledgeBase Cardio Risk
## Subgoal 6 - AI Model Docker Provision (KnowledgeBase)

## Ownership
Muhammad Farhan Tanvir, Md Asifuzzaman Jishan

## Course and Institution
Created as part of the course "M. Grum: Advanced AI-based Application Systems" by the 'Junior Chair for Business Information Science, esp. AI-based Application Systems' at University of Potsdam

## AI Model Characterization

This Docker image provides trained AI models for cardiovascular disease risk prediction:

### 1. Artificial Neural Network (ANN) Model
- **File**: `currentAiSolution.keras`
- **Type**: Deep Learning Classification Model
- **Architecture**: Sequential Neural Network (64→32→1 neurons)
- **Task**: Binary classification of heart disease presence (0/1)
- **Performance**: 71.12% test accuracy
- **Target Variable**: cardio (binary)
- **Features**: 11 cardiovascular indicators (age, gender, blood pressure, cholesterol, etc.)
- **Framework**: TensorFlow/Keras

### 2. Ordinary Least Squares (OLS) Regression Model
- **File**: `currentOlsSolution.pkl`
- **Type**: Linear Regression Model
- **Task**: Continuous cardiovascular risk score prediction (0-1 scale)
- **Performance**: R² = 0.9998, RMSE = 0.0017
- **Target Variable**: risk_score (continuous)
- **Features**: 11 cardiovascular indicators (same as ANN)
- **Framework**: Statsmodels

### Model Comparison
- **ANN**: Best for binary diagnostic decisions (disease/no disease)
- **OLS**: Best for interpretable continuous risk assessment
- **Complementary**: Both models provide different perspectives on cardiovascular risk

## Content / Paths inside the image
- `/tmp/knowledgeBase/currentAiSolution.keras` - Trained ANN model
- `/tmp/knowledgeBase/currentOlsSolution.pkl` - Trained OLS model
- `/tmp/knowledgeBase/README.md` - This documentation

## Data Origin
Models trained on the "Cardiovascular Disease Dataset" by sulianova from Kaggle.
Dataset: [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

## Usage Instructions

**Build Docker Image:**
```bash
docker build -t knowledgebase_cardio_risk ./images/knowledgeBase_cardio_risk
```

**Verify Image Contents:**
```bash
docker run --rm knowledgebase_cardio_risk sh -c "ls -lh /tmp/knowledgeBase"
```

**Tag for Docker Hub:**
```bash
docker tag knowledgebase_cardio_risk jishan900/knowledgebase_cardio_risk:latest
```

**Push to Docker Hub:**
```bash
docker login
docker push jishan900/knowledgebase_cardio_risk:latest
```

**Pull from Docker Hub:**
```bash
docker pull jishan900/knowledgebase_cardio_risk:latest
```

**Docker Compose Test:**
```bash
docker volume create ai_system
docker compose -f images/knowledgeBase_cardio_risk/Docker-compose.yml up --remove-orphans
docker compose -f images/knowledgeBase_cardio_risk/Docker-compose.yml down
```

## License Commitment
AGPL-3.0 license commitment.

This software is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
You are free to use, modify, and distribute this software under the terms of the AGPL-3.0 license.
Any modifications or derivative works must also be licensed under AGPL-3.0 and made available to users.

For more information, see: https://www.gnu.org/licenses/agpl-3.0.en.html