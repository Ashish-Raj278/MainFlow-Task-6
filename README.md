# MainFlow-Task-6

# Heart Disease Prediction using Logistic Regression

## 📌 Overview
This project implements a **Logistic Regression** model to classify patients as having heart disease or not based on medical parameters. The dataset contains various health metrics such as age, cholesterol levels, and blood pressure. The model is built **without using Scikit-Learn**, ensuring a complete understanding of logistic regression from scratch.

---

## 📂 Project Structure
```
📦 Heart Disease Prediction
├── 📁 data                 # Dataset files (Excel or CSV)
├── 📁 src                  # Source code
│   ├── preprocess.py       # Data preprocessing functions
│   ├── train.py            # Logistic Regression model training
│   ├── evaluate.py         # Model evaluation metrics
│   ├── utils.py            # Helper functions
├── README.md               # Project documentation
└── requirements.txt        # Dependencies
```

---

## ⚙️ Features
- **Manual implementation of Logistic Regression** (No Scikit-Learn)
- **Dataset preprocessing** (handling missing values, normalization)
- **Model training and optimization**
- **Performance evaluation** (Confusion matrix, Accuracy, Precision, Recall, F1-score)

---

## 📊 Dataset
The dataset consists of several features such as:
- **Age**
- **Cholesterol Level**
- **Blood Pressure**
- **Other relevant medical parameters**
- **Target Variable**: 1 (Has heart disease) or 0 (No heart disease)

Ensure that the dataset is correctly formatted before running the scripts.

---

## 🚀 Installation
### Prerequisites
- Python 3.x
- Pandas
- NumPy
- Matplotlib (for visualization)

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏗️ Usage
### 1️⃣ Preprocess Data
```bash
python src/preprocess.py
```
This step cleans and normalizes the dataset.

### 2️⃣ Train the Model
```bash
python src/train.py
```
This trains the logistic regression model and saves the parameters.

### 3️⃣ Evaluate the Model
```bash
python src/evaluate.py
```
This outputs accuracy, precision, recall, and F1-score.

---

## 📈 Model Evaluation
### Confusion Matrix
| Actual \ Predicted | Predicted: 1 (Disease) | Predicted: 0 (No Disease) |
|-------------------|---------------------|-----------------|
| **Actual: 1** | **True Positive (TP)** | **False Negative (FN)** |
| **Actual: 0** | **False Positive (FP)** | **True Negative (TN)** |

### Performance Metrics
- **Accuracy** = `(TP + TN) / (TP + TN + FP + FN)`
- **Precision** = `TP / (TP + FP)`
- **Recall** = `TP / (TP + FN)`
- **F1-Score** = `2 * (Precision * Recall) / (Precision + Recall)`

---

## 🎯 Results
| Metric | Score |
|--------|-------|
| Accuracy | 85% |
| Precision | 90.9% |
| Recall | 83.3% |
| F1-Score | 86.9% |

---

## 🤝 Contributing
Feel free to submit issues or pull requests if you’d like to improve the project.

---

## 📜 License
This project is licensed under the MIT License.

---

## ⭐ Acknowledgments
- Thanks to [source of dataset] for providing the dataset.
- Inspired by classic machine learning implementations.

---



