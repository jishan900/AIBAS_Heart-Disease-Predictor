# CodeBase Cardio Risk
## Subgoal 6 - AI Model Docker Provision (CodeBase)

## Ownership
Muhammad Farhan Tanvir, Md Asifuzzaman Jishan

## Course and Institution
Created as part of the course "M. Grum: Advanced AI-based Application Systems" by the 'Junior Chair for Business Information Science, esp. AI-based Application Systems' at University of Potsdam

## AI Model Code Characterization

This Docker image provides the complete source code for cardiovascular disease risk prediction models:

### 1. ANN Training Code
- **File**: `train_ann.py`
- **Purpose**: Trains Artificial Neural Network for binary heart disease classification
- **Features**: 
  - Sequential model with 64→32→1 architecture
  - Binary crossentropy loss, Adam optimizer
  - Early stopping and model checkpointing
  - Comprehensive evaluation with ROC, confusion matrix, precision-recall curves
  - Generates diagnostic visualizations

### 2. OLS Training Code
- **File**: `train_ols.py`
- **Purpose**: Trains Ordinary Least Squares regression for continuous risk prediction
- **Features**:
  - Statsmodels OLS implementation
  - Statistical diagnostic tests (Durbin-Watson, Breusch-Pagan, White test)
  - Residual analysis and normality checks
  - Feature importance visualization
  - Comprehensive diagnostic plots

### 3. Model Comparison Code
- **File**: `model_comparison.py`
- **Purpose**: Compares performance between ANN and OLS models
- **Features**:
  - Side-by-side performance metrics
  - Visualization comparisons
  - Model strengths/weaknesses analysis
  - Recommendations for model usage

### Code Capabilities
- **Data Processing**: Handles cardiovascular dataset preprocessing
- **Model Training**: Implements both neural network and linear regression approaches
- **Evaluation**: Comprehensive performance assessment with statistical tests
- **Visualization**: Generates diagnostic plots, scatter plots, and comparison charts
- **Export**: Saves trained models in appropriate formats (.keras, .pkl)

## Content / Paths inside the image
- `/tmp/codeBase/train_ann.py` - ANN training script
- `/tmp/codeBase/train_ols.py` - OLS training script  
- `/tmp/codeBase/model_comparison.py` - Model comparison script
- `/tmp/codeBase/README.md` - This documentation

## Data Origin
Code designed for the "Cardiovascular Disease Dataset" by sulianova from Kaggle.
Dataset: [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

## Usage Instructions

**Build Docker Image:**
```bash
docker build -t codebase_cardio_risk ./images/codeBase_cardio_risk
```

**Verify Image Contents:**
```bash
docker run --rm codebase_cardio_risk sh -c "ls -lh /tmp/codeBase"
```

**Extract Code Files:**
```bash
docker run --rm -v $(pwd):/output codebase_cardio_risk sh -c "cp /tmp/codeBase/*.py /output/"
```

**Tag for Docker Hub:**
```bash
docker tag codebase_cardio_risk jishan900/codebase_cardio_risk:latest
```

**Push to Docker Hub:**
```bash
docker login
docker push jishan900/codebase_cardio_risk:latest
```

**Pull from Docker Hub:**
```bash
docker pull jishan900/codebase_cardio_risk:latest
```

**Docker Compose Test:**
```bash
docker volume create ai_system
docker compose -f images/codeBase_cardio_risk/Docker-compose.yml up --remove-orphans
docker compose -f images/codeBase_cardio_risk/Docker-compose.yml down
```

## Technical Requirements
- Python 3.8+
- TensorFlow/Keras 2.x
- Statsmodels
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

## License Commitment
AGPL-3.0 license commitment.

This software is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
You are free to use, modify, and distribute this software under the terms of the AGPL-3.0 license.
Any modifications or derivative works must also be licensed under AGPL-3.0 and made available to users.

For more information, see: https://www.gnu.org/licenses/agpl-3.0.en.html