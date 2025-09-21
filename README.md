# Machine Learning Projects

This repository contains multiple machine learning projects completed as part of my coursework and independent study. Each project explores different algorithms, techniques, and evaluation methods.

---

## 🍷 Wine Quality Classification

A project using the **UCI Wine dataset** to predict wine quality with traditional machine learning models.

### Features
- Implemented algorithms **from scratch**: k-NN, Naive Bayes, Logistic Regression (Softmax).
- Benchmarked against scikit-learn models, including Decision Trees.
- Applied **Principal Component Analysis (PCA)** (custom + sklearn).
- Evaluated with accuracy, precision, recall, and F1-score.

### Results
- **Naive Bayes & Logistic Regression** achieved the highest accuracy (~98%).
- **Custom implementations** matched sklearn results, validating correctness.
- PCA improved **Decision Tree** generalisation (accuracy ↑ from 91% → 96%).
- PCA slightly reduced **k-NN** performance (96% → 94%).

### Tech Stack
Python · NumPy · Pandas · scikit-learn · Matplotlib

---

## 🖼 CIFAR-10 Image Classification

A deep learning project on the **CIFAR-10 dataset** using fully connected neural networks.

### Features
- Baseline dense neural network, scaled up with deeper/wider layers.
- Applied **L2 regularisation and dropout** to reduce overfitting.
- **Hyperparameter tuning** with Keras Tuner (Hyperband).
- Evaluated using accuracy, precision, recall, and F1-score.

### Results
- Baseline model: modest accuracy, overfitting.
- Regularised model: dropout=0.1, L2=0.0001 improved generalisation.
- Tuned model: 4 hidden layers (1024 → 256), dropout=0.15, L2≈8e-5 → best results.
- Future improvements: CNNs, data augmentation, ensemble methods.

### Tech Stack
Python · TensorFlow/Keras · NumPy · Pandas · Matplotlib · Keras Tuner

---

## 📖 References
- [UCI Wine Dataset (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)  
- [CIFAR-10 Dataset (TensorFlow Datasets)](https://www.tensorflow.org/datasets/catalog/cifar10)   
- scikit-learn, TensorFlow/Keras, and Keras Tuner Documentation
