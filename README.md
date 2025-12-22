# Cardiovascular Disease Prediction System

An AI-based system for **cardiovascular disease risk analysis** using real-world data from Kaggle. This project implements a complete machine learning pipeline with containerized deployment for AI-CPS (Cyber-Physical Systems) integration.

## 🎯 Project Overview

The system provides dual prediction approaches:
- **🧠 Artificial Neural Network (ANN)**: Binary heart disease classification (0/1)
- **📊 Ordinary Least Squares (OLS)**: Continuous cardiovascular risk score (0-1 scale)

## 🏆 Key Achievements

- **71.12% ANN accuracy** for binary disease classification
- **R² = 0.9998** for OLS continuous risk prediction  
- **4 Docker images** published to Docker Hub for reproducible deployment
- **Complete AI-CPS integration** with external volume mounting
- **Comprehensive model comparison** and performance analysis

## 📊 Dataset Overview

**Source**: [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) by sulianova (Kaggle)

**Features**: 11 cardiovascular indicators
- Demographics: age, gender, height, weight
- Vital signs: systolic/diastolic blood pressure (ap_hi, ap_lo)
- Lab results: cholesterol, glucose levels
- Lifestyle: smoking, alcohol consumption, physical activity

**Targets**:
- `cardio`: Binary heart disease indicator (0/1) - for ANN classification
- `risk_score`: Continuous cardiovascular risk (0-1) - for OLS regression

**Dataset Size**: 70,000 patient records → 62,501 after preprocessing (80/20 train/test split)

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.8+ (for local development)
- Kaggle API credentials (for data download)

### 1. Clone Repository
```bash
git clone https://github.com/jishan900/AIBAS_Heart-Disease-Predictor.git
cd AIBAS_Heart-Disease-Predictor
```

### 2. Pull Docker Images
```bash
docker pull farhantanvirf07/knowledgebase_cardio_risk:latest
docker pull farhantanvirf07/codebase_cardio_risk:latest
docker pull farhantanvirf07/learningbase_cardio_risk:latest
docker pull farhantanvirf07/activationbase_cardio_risk:latest
```

### 3. Create External Volume
```bash
docker volume create ai_system
```

### 4. Run Model Applications
```bash
# ANN Model Application (Binary Classification)
docker compose -f docker-compose-ann-application.yml up --remove-orphans

# OLS Model Application (Continuous Risk Prediction)
docker compose -f docker-compose-ols-application.yml up --remove-orphans
```

## 🏗️ Project Structure

```
AIBAS_Heart-Disease-Predictor/
├── 📁 code/
│   ├── 📁 ANN/
│   │   └── train_ann.py              # Neural network training
│   ├── 📁 OLS/
│   │   ├── train_ols.py              # Linear regression training
│   │   └── model_comparison.py       # Model performance comparison
│   ├── 📁 data-preprocessing/
│   │   └── prepare_dataset.py        # Data cleaning & feature engineering
│   └── 📁 data-scrapping/
│       └── data_scrapping.py         # Kaggle dataset download
├── 📁 data/
│   └── 📁 Cardiovascular_disease_dataset/
│       ├── training_data.csv         # Training set (80%)
│       ├── test_data.csv            # Test set (20%)
│       ├── activation_data.csv      # Single sample for inference
│       └── joint_data_collection.csv # Complete processed dataset
├── 📁 documentation/
│   ├── 📁 ANN_Results/              # Neural network outputs
│   ├── 📁 OLS_Results/              # Linear regression outputs
│   └── 📁 Model_Comparison/         # Performance comparison
├── 📁 images/                       # Docker configurations
│   ├── 📁 knowledgeBase_cardio_risk/    # Trained models
│   ├── 📁 codeBase_cardio_risk/         # Training scripts
│   ├── 📁 learningBase_cardio_risk/     # Datasets
│   └── 📁 activationBase_cardio_risk/   # Inference data
├── docker-compose-ann-application.yml   # ANN model deployment
├── docker-compose-ols-application.yml   # OLS model deployment
└── README.md
```


## 🤖 Machine Learning Models

### 🧠 Artificial Neural Network (ANN)
- **Architecture**: Sequential (64 → 32 → 1 neurons)
- **Activation**: ReLU → ReLU → Sigmoid
- **Task**: Binary heart disease classification
- **Performance**: 71.12% test accuracy
- **Output**: Probability of heart disease (0-1) → Binary prediction (0/1)
- **Framework**: TensorFlow/Keras

**Key Metrics**:
- Precision: 0.78 (class 1), 0.67 (class 0)
- Recall: 0.59 (class 1), 0.83 (class 0)
- F1-Score: 0.67 (class 1), 0.74 (class 0)

### 📊 Ordinary Least Squares (OLS)
- **Type**: Linear regression with statistical analysis
- **Task**: Continuous cardiovascular risk prediction
- **Performance**: R² = 0.9998, RMSE = 0.0017
- **Output**: Risk score (0-1 scale)
- **Framework**: Statsmodels

**Key Features**:
- Highly interpretable coefficients
- Statistical significance testing
- Diagnostic plots (residuals, Q-Q, leverage)
- Feature importance analysis

### 🔄 Model Comparison
| Aspect | ANN | OLS |
|--------|-----|-----|
| **Task** | Binary Classification | Continuous Regression |
| **Target** | cardio (0/1) | risk_score (0-1) |
| **Performance** | 71.12% accuracy | R² = 0.9998 |
| **Interpretability** | Low (black box) | High (linear coefficients) |
| **Use Case** | Diagnostic decisions | Risk assessment |

## 🐳 Docker Deployment

### Available Images
All images are publicly available on Docker Hub:

1. **Knowledge Base**: `farhantanvirf07/knowledgebase_cardio_risk:latest`
   - Contains trained models (ANN + OLS)
   - Size: ~10 MB

2. **Code Base**: `farhantanvirf07/codebase_cardio_risk:latest`
   - Contains training scripts and model comparison
   - Size: ~40 KB

3. **Learning Base**: `farhantanvirf07/learningbase_cardio_risk:latest`
   - Contains training and validation datasets
   - Size: Variable (depends on dataset)

4. **Activation Base**: `farhantanvirf07/activationbase_cardio_risk:latest`
   - Contains single sample for model inference
   - Size: ~5 KB

### Docker Compose Applications

#### ANN Model Application
```bash
docker compose -f docker-compose-ann-application.yml up --remove-orphans
```
**Output**: Binary heart disease prediction with confidence score

#### OLS Model Application  
```bash
docker compose -f docker-compose-ols-application.yml up --remove-orphans
```
**Output**: Continuous cardiovascular risk score (0-1 scale)

### Volume Integration
- **External Volume**: `ai_system` for data persistence
- **Mount Point**: `/mnt` (avoids container `/tmp` conflicts)
- **Workflow**: Sequential loading → Model application → Results

## 📈 Results & Performance

### ANN Model Results
- **Test Accuracy**: 71.12%
- **Training Accuracy**: 64.94%
- **Validation Accuracy**: 69.11%
- **Model File**: `currentAiSolution.keras` (58.7 KB)

### OLS Model Results  
- **Test R²**: 0.9998
- **Test RMSE**: 0.0017
- **Training R²**: 0.9999
- **Model File**: `currentOlsSolution.pkl` (9.9 MB)

### Visualizations Generated
- Training/validation curves
- ROC curves and AUC analysis
- Confusion matrices
- Precision-recall curves
- Residual analysis plots
- Feature importance charts

## 🔬 Data Processing Pipeline

### 1. Data Acquisition
```bash
python code/data-scrapping/data_scrapping.py
```
- Downloads from Kaggle API
- Extracts cardiovascular dataset
- Creates `joint_data_collection.csv`

### 2. Data Preprocessing
```bash
python code/data-preprocessing/prepare_dataset.py
```
- Removes duplicates and missing values
- Outlier detection using IQR method
- Min-max normalization (0-1 scale)
- BMI calculation and feature engineering
- Risk score computation using weighted features
- 80/20 train/test split

### 3. Model Training
```bash
# Train ANN model
python code/ANN/train_ann.py

# Train OLS model  
python code/OLS/train_ols.py

# Compare models
python code/OLS/model_comparison.py
```

## 🎯 Use Cases & Applications

### Clinical Decision Support
- **ANN**: Binary diagnostic screening (Disease/No Disease)
- **OLS**: Continuous risk stratification for patient monitoring

### AI-CPS Integration
- **IoT Sensors**: Real-time vital sign monitoring
- **Edge Computing**: Local model inference on medical devices
- **MQTT Communication**: Integration with hospital information systems

### Research Applications
- **Model Comparison**: ANN vs. traditional statistical methods
- **Feature Analysis**: Understanding cardiovascular risk factors
- **Performance Benchmarking**: Baseline for future improvements

## 🛠️ Development Setup

### Local Development
1. **Install Dependencies**:
   ```bash
   pip install tensorflow pandas scikit-learn statsmodels matplotlib seaborn
   ```

2. **Set up Kaggle API**:
   ```bash
   pip install kaggle
   # Place kaggle.json in ~/.kaggle/
   ```

3. **Run Complete Pipeline**:
   ```bash
   # Download and preprocess data
   python code/data-scrapping/data_scrapping.py
   python code/data-preprocessing/prepare_dataset.py
   
   # Train models
   python code/ANN/train_ann.py
   python code/OLS/train_ols.py
   
   # Compare performance
   python code/OLS/model_comparison.py
   ```

### Docker Development
1. **Build Images Locally**:
   ```bash
   docker build -t knowledgebase_cardio_risk ./images/knowledgeBase_cardio_risk
   docker build -t codebase_cardio_risk ./images/codeBase_cardio_risk
   docker build -t learningbase_cardio_risk ./images/learningBase_cardio_risk
   docker build -t activationbase_cardio_risk ./images/activationBase_cardio_risk
   ```

2. **Test Individual Images**:
   ```bash
   docker volume create ai_system
   docker compose -f images/knowledgeBase_cardio_risk/Docker-compose.yml up
   ```

## 📋 Requirements

### System Requirements
- **OS**: Windows, macOS, Linux
- **Docker**: 20.10+ with Docker Compose
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ free space

### Python Dependencies
```
tensorflow>=2.8.0
pandas>=1.3.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
matplotlib>=3.5.0
seaborn>=0.11.0
numpy>=1.21.0
```

## 🤝 Contributing

### Project Team
- **Muhammad Farhan Tanvir**
- **Md Asifuzzaman Jishan**

### Course Information
Created as part of **"M. Grum: Advanced AI-based Application Systems"**  
**Institution**: Junior Chair for Business Information Science, esp. AI-based Application Systems  
**University**: University of Potsdam

### Development Guidelines
1. Follow existing code structure and naming conventions
2. Add comprehensive documentation for new features
3. Include unit tests for new functionality
4. Update Docker images when modifying core components
5. Maintain AGPL-3.0 license compliance

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

**Key Points**:
- ✅ Free to use, modify, and distribute
- ✅ Must share source code of modifications
- ✅ Network use triggers copyleft requirements
- ✅ Commercial use permitted with compliance

For full license text, see: https://www.gnu.org/licenses/agpl-3.0.en.html

## 📚 Documentation

### Generated Reports
- `documentation/ANN_Results/final_metrics.txt` - Neural network performance
- `documentation/OLS_Results/ols_summary.txt` - Linear regression analysis  
- `documentation/Model_Comparison/comparison_report.txt` - Model comparison
- `documentation/DOCKER_DEPLOYMENT_REPORT.md` - Container deployment guide
- `documentation/SUBGOAL_7_COMPLETION_REPORT.md` - Docker Compose applications

### Visualizations
- Training/validation curves
- ROC and precision-recall curves
- Confusion matrices and diagnostic plots
- Feature importance and coefficient analysis
- Model comparison charts

## 🔗 References

### Data Source
- **Dataset**: [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- **Author**: sulianova
- **Platform**: Kaggle
- **License**: Database Contents License (DbCL) v1.0

### Academic Context
This project demonstrates practical implementation of AI-CPS concepts including:
- Containerized machine learning deployment
- Multi-model comparison and analysis
- External volume integration for data persistence
- Reproducible research through Docker Hub distribution

### Related Work
- Grum M. 2022. Construction of a Concept of Neuronal Modeling. Springer Gabler Wiesbaden.
- Grum, M. et al. 2023. AI Case-Based Reasoning for Artificial Neural Networks. A2IA 2023 Conference.

---

## 🚀 Quick Commands Reference

```bash
# Complete Docker deployment
docker volume create ai_system
docker pull farhantanvirf07/knowledgebase_cardio_risk:latest
docker pull farhantanvirf07/codebase_cardio_risk:latest
docker pull farhantanvirf07/learningbase_cardio_risk:latest
docker pull farhantanvirf07/activationbase_cardio_risk:latest

# Run ANN application
docker compose -f docker-compose-ann-application.yml up --remove-orphans

# Run OLS application  
docker compose -f docker-compose-ols-application.yml up --remove-orphans

# Clean up
docker compose -f docker-compose-ann-application.yml down
docker compose -f docker-compose-ols-application.yml down
```

**For questions or support, please refer to the documentation in the `documentation/` directory.**