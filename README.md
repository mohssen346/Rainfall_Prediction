# Rainfall Intensity Clustering & Classification  
**A Complete Machine Learning Pipeline for Daily Rainfall Prediction**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)  
![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange)  
![Pandas](https://img.shields.io/badge/Pandas-Latest-brightgreen)  
![License](https://img.shields.io/badge/License-MIT-yellow)  
![Status](https://img.shields.io/badge/Project%20Status-Completed-success)

## 📖 Project Overview

This project analyzes long-term daily meteorological data with a focus on **24-hour accumulated precipitation (rrr24)**. The pipeline performs three main tasks:

1. **Data Preprocessing** – Merging multiple historical CSV files into a single dataset  
2. **Unsupervised Clustering** – Grouping rainfall amounts into 5 meaningful intensity classes using KMeans  
3. **Supervised Classification** – Predicting the rainfall intensity class for the year **2017** using data from all previous years with several machine learning models  

The best-performing model is a **Soft Voting Classifier** combining **Random Forest** and **MLP (Neural Network)**, achieving **>92% accuracy** on the 2017 test set.

## 📁 Project Structure

```
Rainfall-Clustering-and-Classification/
├── data/
│   ├── raw/                     # ← Put your original CSV files here
│   ├── combined_output.csv      # Output of preprocessing
│   └── clustered_data.csv       # Output of clustering
├── results/
│   ├── heatmap_plot.png
│   ├── prediction_results.csv
│   └── prediction_results_voting.csv
├── src/
│   ├── 01_preprocess.py         # Merge raw CSV files
│   ├── 02_clustering.py         # KMeans clustering (5 classes)
│   └── 03_classification.py     # Train & evaluate multiple models + Voting
├── requirements.txt
└── README.md
```

## 🚀 How to Run the Project (Step by Step)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Rainfall-Clustering-and-Classification.git
cd Rainfall-Clustering-and-Classification

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate    # Linux/Mac
# or
venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your raw daily weather CSV files inside data/raw/

# 5. Run the pipeline
python src/01_preprocess.py      # Merge all CSVs
python src/02_clustering.py      # Create 5 rainfall intensity clusters
python src/03_classification.py  # Train models & predict 2017
```

After execution you will find:
- Correlation heatmap in `results/`
- Final predictions in `results/prediction_results_voting.csv`

## 🧪 Methodology

### 1. Preprocessing
- All CSV files in `data/raw/` are concatenated
- Duplicates and obvious errors are preserved (real weather data often contains them)

### 2. Clustering (Unsupervised)
- Extreme outliers (>500 mm) removed
- Rainfall values standardized
- KMeans with **k = 5** clusters
- Clusters reordered from lowest to highest rainfall → labels 0–4  
  (0 = No/Light, 1 Light, 2 Moderate, 3 Heavy, 4 Very Heavy/Extreme)

### 3. Classification (Supervised)
- **Training data**: All years before 2017  
- **Test data**: Entire year 2017 (temporal split to avoid leakage)  
- Models evaluated models:
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Gaussian Naive Bayes
  - Multi-Layer Perceptron (MLP)
  - **Soft Voting Classifier (Random Forest + MLP)** ← **Best model**

## 📊 Example Results (2017 Test Set)

| Model                        | Accuracy |
|------------------------------|----------|
| Random Forest                | 0.914    |
| MLP                          | 0.908    |
| SVM                          | 0.876    |
| KNN                          | 0.859    |
| Naive Bayes                  | 0.742    |
| **Voting (RF + MLP)**        | **0.926** |

## 📈 Visualization Example

![Correlation Heatmap](results/heatmap_plot.png)

## 📄 Input Data Format (Expected Columns)

Your raw CSV files should at least contain:
```
year, month, day, rrr24, [other meteorological variables...]
```
Additional columns (temperature, pressure, humidity, wind, etc.) are used as features for classification.

## 🔧 Requirements

All dependencies are listed in `requirements.txt`:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
imbalanced-learn
```

Install with:
```bash
pip install -r requirements.txt
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to open a pull request or an issue.


## ⭐ Show Your Support

Give a ⭐ if this project helped you or if you found it interesting!

Happy forecasting! 🌧️⚡

---
Last updated: November 2025
