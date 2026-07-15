# PhishDetector: End-to-End Phishing Website Detection System

## 🎓 Final Project — Data Science in Cybersecurity

This repository serves as the final project submission for the **Data Science in Cybersecurity** course (Dr. Uri Itai).

### Project Description
This project performs a **critical evaluation and empirical reproduction** of the [Phishing Website Detection](https://github.com/sujeetgund/phishing-website-detection) repository by Sujeet Gund. We faithfully reproduced the author's 5-model cross-validation experiment, then extended it with thorough EDA, feature engineering experiments, a rigorous 5-model comparison with comprehensive metrics, threshold optimization, and detailed false-positive/false-negative error analysis.

### Links
- **Selected Source / Original Repository:** [sujeetgund/phishing-website-detection](https://github.com/sujeetgund/phishing-website-detection)
- **Original GitHub Repository:** [https://github.com/sujeetgund/phishing-website-detection](https://github.com/sujeetgund/phishing-website-detection)
- **Dataset Source:** [UCI ML Repository — Phishing Websites](https://archive.ics.uci.edu/dataset/327/phishing+websites)

### Submission Files
| File | Description |
|---|---|
| 📄 `Report.pdf` | Complete English PDF report with methodology, results, criticism, and conclusions |
| 📓 `project_notebook.ipynb` | Executable Jupyter notebook with all empirical code |
| 🐍 `generate_analysis.py` | Supporting Python script that generates all figures and metrics |
| 📁 `images/` | All generated figures (16 plots) |
| 📊 `metrics.json` | Full evaluation metrics for all 5 models |
| 📊 `reproduction_results.json` | Author vs. our reproduction comparison |

### Execution Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/amaraqusai/datascience.git
   cd datascience
   ```
2. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn nbformat
   ```
3. **Run the analysis script** (generates all figures and metrics):
   ```bash
   python generate_analysis.py
   ```
4. **Open the notebook** in Jupyter, VS Code, or Google Colab:
   ```bash
   jupyter notebook project_notebook.ipynb
   ```

### Dataset Source
The dataset is from the [UCI Machine Learning Repository — Phishing Websites Data Set](https://archive.ics.uci.edu/dataset/327/phishing+websites). It contains 11,055 instances with 30 ordinal features extracted from website URLs and metadata. The target variable indicates whether a website is Phishing (1) or Legitimate (-1).

---

## Original Project Information

The original repository by Sujeet Gund implements an end-to-end ML pipeline for phishing detection using scikit-learn, FastAPI, and Docker. See the original [README](https://github.com/sujeetgund/phishing-website-detection/blob/main/README.md) for full details.

### Tech Stack
* **Language:** Python 3.10+
* **Libraries:** pandas, scikit-learn, matplotlib, seaborn, fastapi
* **Models:** Random Forest, SVM, KNN, Logistic Regression, Ridge, Gradient Boosting
* **API:** FastAPI
* **Packaging:** Docker
