---

## **🛡️ Hybrid DGA Detection System**  
🚀 **Advanced Deep Learning & Graph-Based Approach for Detecting Domain Generation Algorithm (DGA) Domains**  

![DGA Detection](https://upload.wikimedia.org/wikipedia/commons/9/9c/Example_image.jpg) *(Replace with your own image related to cybersecurity or DGA detection.)*

---

## **📌 Table of Contents**
- [🔍 Introduction](#-introduction)
- [📖 Background](#-background)
- [🎯 Project Objectives](#-project-objectives)
- [🛠️ Methodology](#-methodology)
- [📊 Dataset](#-dataset)
- [🏗️ Model Architecture](#-model-architecture)
- [💻 Installation](#-installation)
- [🚀 Usage](#-usage)
- [📈 Results](#-results)
- [📢 Contributing](#-contributing)
- [📜 License](#-license)

---

## **🔍 Introduction**
**Domain Generation Algorithms (DGA)** are used by malware to generate large numbers of domain names dynamically, making it difficult for traditional security systems to block them. This project aims to **build an advanced hybrid AI model** that effectively detects and classifies **DGA-generated domains** using **Deep Learning (CNN, BiLSTM, GPT)** and **Graph Neural Networks (GCN)**.

---

## **📖 Background**
🔹 Traditional **signature-based detection** fails to detect **zero-day** DGA domains.  
🔹 **Machine learning & deep learning** techniques have **improved detection** but often fail with **new DGA families**.  
🔹 **Graph Neural Networks (GCN) & NLP models (GPT)** can capture domain structures more effectively.

---

## **🎯 Project Objectives**
✅ **Accurate DGA Detection**: Build a model that outperforms traditional classifiers.  
✅ **Generalization to Zero-Day DGAs**: Leverage NLP-based models like **GPT** for enhanced feature learning.  
✅ **Graph-Based Analysis**: Apply **Graph CNN (GCN)** to detect structural relationships between domains.  
✅ **Robust Preprocessing Pipeline**: Handle raw datasets efficiently, ensuring proper cleaning and balancing.  

---

## **🛠️ Methodology**
We employ a **hybrid deep learning and machine learning pipeline**:
1. **Data Preprocessing**  
   - **Cleaning** raw domain lists.  
   - **Tokenization** (Character & Word-Level).  
   - **Feature Extraction** (TF-IDF, FastText, GPT Embeddings).  
2. **Graph Construction**  
   - Convert domain datasets into a **graph representation**.  
   - Apply **Levenshtein Distance** for node connections.  
3. **Model Training**  
   - **Graph CNN (GCN)** for relationship learning.  
   - **GPT-based Transformer** for contextual domain analysis.  
   - **Support Vector Machine (SVM)** as a final classifier.  

---

## **📊 Dataset**
We use a combination of **legitimate and malicious domains**:  
| Dataset | Source | Size | Label |  
|----------|----------|------|------|  
| **Alexa** | Alexa Top 1M | 1M | Legitimate (0) |  
| **UMUDGA** | 50+ DGA Families | 500K | DGA (1) |  
| **360NetLab** | Malware Analysis | 337K | DGA (1) |  

---

## **🏗️ Model Architecture**
**Hybrid Approach:**
✅ **GCN (Graph CNN)** → Learns structural relationships between domains.  
✅ **GPT (NLP-Based Model)** → Captures sequence-level patterns in domain names.  
✅ **LLN (Logistic Label Normalization)** → Improves classification generalization.  
✅ **SVM (Final Classifier)** → Robust decision-making based on extracted features.  

![Model Architecture](https://upload.wikimedia.org/wikipedia/commons/3/3c/Neural_Network_Model.jpg) *(Replace with your own model diagram.)*

---

## **💻 Installation**
### **🔹 Prerequisites**
Ensure you have **Python 3.8+** and the following dependencies installed:

```bash
pip install torch transformers torch-geometric scikit-learn pandas networkx matplotlib fasttext imbalanced-learn
```

### **🔹 Clone the Repository**
```bash
git clone https://github.com/zenbenali/HSDGA-NLP
cd DGA-Detection-Hybrid
```

---

## **🚀 Usage**
### **🔹 1. Data Preprocessing**
Run the preprocessing script to clean and balance datasets.
```bash
python preprocess.py --input data/ --output processed_data/
```

### **🔹 2. Train the Model**
Train the hybrid model with GCN + GPT + SVM.
```bash
python train.py --epochs 20 --batch_size 32 --use_gcn --use_gpt
```

### **🔹 3. Evaluate the Model**
Run evaluation scripts to get **accuracy, precision, recall, and AUC scores**.
```bash
python evaluate.py --input test_data/
```

### **🔹 4. Real-Time Prediction**
To predict if a domain is DGA-generated:
```python
from model import predict_dga

domain = "exampledga.xyz"
prediction = predict_dga(domain)
print(f"Domain {domain} is {'DGA' if prediction == 1 else 'Legitimate'}")
```

---

## **📈 Results**
| Model | Accuracy (%) | AUC Score | Best For |  
|------------|-------------|------------|------------|  
| **Random Forest** | 92.3% | 0.91 | Baseline |  
| **CNN + BiLSTM** | 97.7% | 0.96 | DGA Detection |  
| **GCN + GPT + SVM (Ours)** | **98.8%** | **0.98** | Generalization & Zero-Day DGAs |  

📌 **Key Findings:**
✅ **GCN captures structural similarities**, improving classification.  
✅ **GPT embeddings generalize well** for **previously unseen DGA families**.  
✅ **Hybrid model outperforms traditional ML models**.  

---

## **📢 Contributing**
We welcome contributions! To contribute:
1. **Fork the repository**.
2. **Create a new branch**:  
   ```bash
   git checkout -b feature-branch
   ```
3. **Commit your changes** and **push to GitHub**:  
   ```bash
   git commit -m "Added new feature"
   git push origin feature-branch
   ```
4. **Submit a Pull Request (PR).**

---

## **📜 License**
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## **🌟 Acknowledgments**
Special thanks to:
- **UMUDGA & 360NetLab** for providing **DGA datasets**.
- **Researchers in AI & Cybersecurity** for inspiration.

---

### **📌 Want to Improve DGA Detection?**
🔗 [**Check out the full project on GitHub!**](https://github.com/zenbenali/HSDGA-NLP)  

🚀 **Star this repo if you find it useful!** ⭐

---

### **📌 Final Notes**

