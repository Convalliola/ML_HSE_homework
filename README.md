Этот репозиторий содержит набор домашних заданий по машинному обучению, выполненных в рамках курса ВШЭ, материалы которого представлены на YouTube и в репозитории Github https://github.com/esokolov/ml-course-hse/tree/master/2019-fall . Проект демонстрирует практическую реализацию фундаментальных алгоритмов машинного обучения, методов оптимизации и методов предварительной обработки данных с использованием Python и популярных библиотек машинного обучения.

## Информация о курсе

- Курс: Машинное обучение
- Образовательная организация: ФКН ВШЭ
- Формат: курс на YouTube (запись лекций и семинаров) с практическими заданиями

## Обзор репозитория

### 1. **Gradient Descent Implementation** (`homework_practice_03.ipynb`)
**Topics**: Optimization Algorithms, Linear Regression

- **Full Gradient Descent**: Implementation of batch gradient descent for linear regression
- **Stochastic Gradient Descent**: Mini-batch optimization with customizable batch sizes
- **Momentum Method**: Advanced optimization technique with momentum coefficient
- **Key Features**:
  - Vectorized computations for efficiency
  - Convergence criteria with tolerance and max iterations
  - Loss history tracking for convergence analysis
  - Customizable learning rates and momentum parameters

### 2. **Support Vector Machines & Probability Calibration** (`homework_practice_04.ipynb`)
**Topics**: SVM, Probability Calibration, Categorical Encoding

- **Support Vector Machines**:
  - Linear SVM implementation and visualization
  - Support vector identification and visualization
  - ROC-AUC and PR-AUC analysis
- **Probability Calibration**:
  - Calibration curves for logistic regression vs SVM
  - `CalibratedClassifierCV` implementation
  - Sigmoid and isotonic calibration methods
- **Categorical Variable Processing**:
  - One-hot encoding implementation
  - Target encoding (mean encoding) with smoothing
  - Comparison of encoding methods' performance
  - Overfitting prevention techniques

### 3. **Decision Trees & Ensemble Methods** (`homework_practice_05.ipynb`)
**Topics**: Decision Trees, Ensemble Learning, Categorical Features

- **Decision Tree Analysis**:
  - Visualization of decision boundaries
  - Hyperparameter tuning and analysis
  - Custom decision tree implementation
- **Dataset Analysis**:
  - Synthetic datasets (moons, circles, classification)
  - Real-world categorical data processing
  - Performance comparison across different encoding methods

### 4. **Bias-Variance Decomposition** (`homework_practice_06.ipynb`)
**Topics**: Model Evaluation, Bootstrap Methods, Statistical Analysis

- **Bootstrap Implementation**:
  - Bias and variance estimation using bootstrap sampling
  - Error decomposition analysis
  - Model complexity vs performance trade-offs
- **Statistical Analysis**:
  - Mathematical expectation estimation
  - Variance analysis across different algorithms
  - Model selection based on bias-variance trade-off

### 5. **Advanced ML Topics** (`homework_practice_07.ipynb`)
**Topics**: Advanced Machine Learning Concepts

*[Content details to be added based on file analysis]*

### 6. **Additional Practice** (`hw_practice_02.ipynb`)
**Topics**: Foundational ML Concepts

*[Content details to be added based on file analysis]*

## 🛠️ Technical Stack

- **Python**: Primary programming language
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and utilities
- **Matplotlib/Seaborn**: Data visualization
- **SciPy**: Scientific computing functions



## 📁 Repository Structure

```
ML_homework/
├── homework_practice_03.ipynb    # Gradient Descent Implementation
├── homework_practice_04.ipynb    # SVM & Probability Calibration
├── homework_practice_05.ipynb    # Decision Trees & Ensembles
├── homework_practice_06.ipynb    # Bias-Variance Decomposition
├── homework_practice_07.ipynb    # Advanced ML Topics
├── hw_practice_02.ipynb          # Foundational ML Concepts
└── README.md                     # This file
```

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone [repository-url]
   cd ML_homework
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn scipy
   ```

3. **Run Jupyter notebooks**:
   ```bash
   jupyter notebook
   ```

4. **Start with any notebook** - Each notebook is self-contained with clear explanations and examples.

## 📈 Performance Highlights

- **Gradient Descent**: Achieved convergence with custom tolerance and iteration controls
- **SVM Implementation**: ROC-AUC of 0.93 on synthetic datasets
- **Categorical Encoding**: Improved model performance from 0.625 to 0.643 AUC using target encoding
- **Feature Selection**: Successfully reduced feature dimensionality while maintaining model performance
- **Probability Calibration**: Significantly improved probability estimates using calibration techniques


