# PhishDetector: End-to-End Phishing Website Detection System

## 🎓 Final Project Submission

This repository serves as the final project submission for the **Data Science in Cyber** course.

- **Project Description**: This project evaluates and reproduces an end-to-end machine learning solution designed to detect phishing websites using URL and metadata features. We verified the original author's claims that a Random Forest Classifier is optimal for this specific tabular dataset format, achieving ~97.1% accuracy, and conducted our own Exploratory Data Analysis, Feature Engineering Analysis, and Error Analysis.
- **Link to the Selected Source / Original GitHub Repository**: [sujeetgund/phishing-website-detection](https://github.com/sujeetgund/phishing-website-detection)
- **Dataset Source**: [UCI Machine Learning Repository - Phishing Websites Data Set](https://archive.ics.uci.edu/dataset/327/phishing+websites)
- **Execution Instructions**: 
  1. Ensure you have Python installed.
  2. Install dependencies: `pip install -r requirements.txt`
  3. To execute the reproduction notebook: Run `project_notebook.ipynb` in Jupyter Notebook, VS Code, or Google Colab.
  4. (Optional) To run the original API server: `uvicorn run_api:app`

The required files for this submission are located in the root of this repository:
- 📄 **PDF report**: `Report.pdf`
- 📓 **Python notebook**: `project_notebook.ipynb`
- 📁 **Supporting code**: `generate_plots.py` and the `src/` directory.

---

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/sujeetgund/phishing-website-detection)](https://github.com/sujeetgund/phishing-website-detection/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/sujeetgund/phishing-website-detection)](https://github.com/sujeetgund/phishing-website-detection/releases)

PhishDetector is an end-to-end machine learning solution that detects phishing websites using URL and metadata features. It integrates data ingestion, validation, preprocessing, model training, and real-time inference into a seamless pipeline, enabling scalable and accurate phishing detection. The solution is fully containerized with Docker and offers a FastAPI-based inference API for easy deployment and integration.

## 🧐 Problem Statement
Phishing websites are fraudulent sites designed to mimic legitimate ones, aiming to steal sensitive information from unsuspecting users. Detecting these websites in real-time is crucial to protect users from scams, data breaches, and financial losses.

The challenge lies in accurately distinguishing phishing websites from legitimate ones using measurable features derived from the site’s URL and metadata. This project addresses this challenge by leveraging machine learning techniques to build a robust phishing detection system.


## 🎯 Objectives
- ✅ Ingest and validate the Phishing Websites dataset.
- ✅ Preprocess and transform features for machine learning readiness.
- ✅ Train multiple ML models (Random Forest, SVC, KNN, Logistic Regression, Ridge) and identify the best-performing one.
- ✅ Evaluate the best model on unseen data and document its performance.
- ✅ Deploy an API using FastAPI for real-time predictions on new website data.
- ✅ Provide clear documentation and modular code for reproducibility and future improvements.

## 📊 Dataset

- **Source**: [UCI Machine Learning Repository - Phishing Websites Data Set](https://archive.ics.uci.edu/dataset/327/phishing+websites)
- **File**: `data/raw/phishingData.csv`
- **Features**: 30 website attributes (e.g., URL length, presence of IP, HTTPS usage, etc.)
- **Target**: `Result` column (1 = Phishing, -1 = Legitimate)
- **Feature Extraction Guide**:
    `docs/Phishing_Websites_Features.pdf` – explains how the dataset’s features were derived.


## 📈 Results & Performance
After extensive experimentation with multiple models, here’s a summary of their performance:

| Model        | Mean Fit Time (s) | Mean Test Score | Std Test Score | Best Estimator                                       |
| ------------ | ----------------: | --------------: | -------------: | ---------------------------------------------------- |
| RandomForest |              1.02 |          0.9711 |         0.0041 | `RandomForestClassifier(random_state=42)`            |
| SVC          |              5.03 |          0.9629 |         0.0064 | `SVC(probability=True, random_state=42)`             |
| KNeighbors   |              0.01 |          0.9623 |         0.0046 | `KNeighborsClassifier()`                             |
| Logistic     |              0.10 |          0.9270 |         0.0047 | `LogisticRegression(max_iter=1000, random_state=42)` |
| Ridge        |              0.01 |          0.9206 |         0.0053 | `RidgeClassifier(random_state=42)`                   |

The Random Forest model demonstrated the best performance, achieving ~97.1% accuracy with high stability across validation folds.

Evaluation and training reports are stored in:
- 📄 [artifacts/reports/evaluation_report.yaml](artifacts/reports/evaluation_report.yaml)
- 📄 [artifacts/reports/training_report.yaml](artifacts/reports/training_report.yaml)

The final trained model is stored in:
`artifacts/models/model.pkl`


## 🌳 Repository Structure

```bash
.
├── Dockerfile
├── LICENSE
├── README.md
├── requirements.txt
├── run_api.py
├── setup.py
├── main.py
├── data/
│   ├── phishingData.csv
│   └── schema.yaml
├── docs/
│   └── Phishing_Websites_Features.pdf
├── notebooks/
├── artifacts/
│   ├── feature_store/
│   │   ├── ingested/{train.csv, test.csv}
│   │   └── validated/{validated_train.csv, validated_test.csv}
│   ├── models/model.pkl
│   └── reports/
│       ├── data_validation_report.yaml
│       ├── evaluation_report.yaml
│       └── training_report.yaml
└── src/phishdetector/
    ├── api/
    │   └── v1/endpoints.py           
    ├── components/
    │   ├── data_ingestion.py
    │   ├── data_validation.py
    │   ├── model_evaluation.py
    │   ├── model_prediction.py
    │   └── model_training.py
    ├── config/
    ├── entity/
    │   ├── artifact_entity.py
    │   └── config_entity.py
    ├── pipelines/
    │   ├── inference/infer.py
    │   ├── preprocessing/preprocess.py
    │   └── training/train_and_evaluate
    └── utils/
        ├── api.py
        ├── common.py
        └── core.py
```

## 📦 Tech Stack

* **Language**: Python 3.10+
* **Libraries**: pandas, scikit-learn, fastapi
* **Models**: Random Forest, Support Vector Machines, KNeighbours, Logistic Regression, and Ridge
* **API**: FastAPI
* **Packaging**: Docker



## ⚙️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/sujeetgund/phishing-website-detection.git
cd phishing-website-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run API Server

```bash
uvicorn run_api:app
```



## 🐳 Docker Usage

### Build the Image

```bash
docker build -t phishdetector .
```

### Run the Container

```bash
docker run -p 8000:8000 phishdetector
```



## 📆 Project Roadmap

* ✅ Dataset ingestion and validation
* ✅ Feature preprocessing and model training
* ✅ Inference pipeline
* ✅ API integration
* ❎ Add Streamlit frontend for usability
* ❎ CI/CD integration with GitHub Actions



## 🤝 Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Commit and push (`git push origin feature-branch`)
5. Open a Pull Request



## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



## ℹ️ Author
If you have questions, suggestions, opportunities, or encounter any issues with this repository, feel free to reach out to **Sujeet Gund**.

[LinkedIn](https://linkedin.com/in/sujeetgund) • [Email](mailto:sujeetgund@gmail.com) • [X (formerly Twitter)](https://x.com/Sujeet_Gund)

