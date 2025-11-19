---

# **🛡️ Hybrid DGA Detection with FastText, CNN, BiLSTM, and Multihead Attention**  
🚀 **Advanced Deep Learning Approach for Detecting Domain Generation Algorithm (DGA) Domains**  

![DGA Detection](https://upload.wikimedia.org/wikipedia/commons/9/9c/Example_image.jpg) *(image related to cybersecurity or DGA detection.)*


## **📌 Table of Contents**
- [🔍 Introduction](#-introduction)
- [📊 Dataset](#-dataset)
- [🏗️ Model Architecture](#-model-architecture)
- [🛠️ Installation](#-installation)
- [🚀 Usage](#-usage)
- [📈 Results](#-results)
- [📢 Contributing](#-contributing)
- [📜 Citation](#-citation)
- [📜 License](#-license)

---
📚 Motivation

DGAs are used by malware to generate many domain names dynamically, helping them evade blacklists. 
arXiv

Pure character-level or statistical models often struggle with “wordlist-based” DGAs, where domain names are built by concatenating dictionary words.

A hybrid NLP model can leverage semantic information (from words) and structural patterns (from characters) to improve detection accuracy.
## **🔍 Introduction**  
Malware often uses **Domain Generation Algorithms (DGAs)** to evade detection by dynamically generating domain names. Traditional detection methods struggle against **zero-day threats**, making **deep learning-based solutions** essential.  

This project builds a **hybrid deep learning model** combining:  
✅ **FastText** for word embeddings  
✅ **CNN** for feature extraction  
✅ **BiLSTM** for sequence modeling  
✅ **Multihead Attention** for focusing on key domain patterns  

This model effectively distinguishes **DGA-generated domains** from **legitimate ones**, achieving high accuracy and robustness.

---

## **📊 Dataset**  
We use a balanced dataset of **1 million domains**:  
| Dataset | Source | Size | Label |  
|----------|----------|------|------|  
| **Legitimate Domains** | Alexa Top 1M | 1 Million | Legitimate (0) |  
| **DGA Domains** | UMUDGA Dataset| 500K | Malicious (1) |  
| **DGA Domaıns** | NetLab360 Dataset| 500K | Malicious (1) |  

**📌 DGA Source:**  
Zago, Mattia; Gil Pérez, Manuel; Martínez Pérez, Gregorio (2020), “UMUDGA - University of Murcia Domain Generation Algorithm Dataset”, *Mendeley Data*, V1, [doi:10.17632/y8ph45msv8.1](https://doi.org/10.17632/y8ph45msv8.1)  

---

## **🏗️ Model Architecture**  
Our hybrid model follows this pipeline:  

1️⃣ **FastText Embeddings** – Convert domain names into meaningful vector representations.  
2️⃣ **CNN Layer** – Extracts spatial patterns from character sequences.  
3️⃣ **BiLSTM Layer** – Captures sequential dependencies and long-range context.  
4️⃣ **Multihead Attention** – Focuses on important character sequences.  
5️⃣ **Dense Layer** – Outputs the probability of a domain being DGA or legitimate.  

**Architecture Diagram:**  
```plaintext
Input → FastText Embeddings → CNN → BiLSTM → Multihead Attention → Fully Connected → Output

```

✅ Why This Approach Works Well

Semantic Strength: By using word-level embeddings, the model understands meaning in domain parts (e.g., “bank”, “login”, “secure”), which is useful for DGA families that use dictionary words.

Structural Insight: Character-level modeling captures patterns like unusual letter combinations, length, or randomness.

Hybrid Power: Combining both gives better detection performance than using either alone.

## **🛠️ Installation**  
### **🔹 Prerequisites**  
Ensure you have Python 3.8+ and the following dependencies installed:  
```bash
pip install tensorflow torch torchvision torchaudio torchtext seaborn scikit-learn pandas numpy matplotlib tqdm
```

### **🔹 Clone the Repository**  
```bash
git clone https://github.com/zenbenali/Hybrid-DGA-Detection.git
cd Hybrid-DGA-Detection
```

---

## **🚀 Usage**  
### **🔹 1. Train the Model**  
```python
python train.py --epochs 5 --batch_size 256
```
The model was trained for **5 epochs** due to time constraints, achieving **97.30% accuracy**.  

### **🔹 2. Evaluate the Model**  
```python
python evaluate.py
```
It calculates **accuracy, precision, recall, F1-score, ROC-AUC**, and more.  

### **🔹 3. Predict a Single Domain**  
To predict whether a domain is **DGA-generated** or **legitimate**:  
```python
from model import predict_dga

domain = "exampledga.xyz"
prediction = predict_dga(domain)
print(f"Domain {domain} is {'DGA' if prediction == 1 else 'Legitimate'}")
```

---

## **📈 Results**  
The model achieved **high performance** on the test set:  
| Metric | Score |  
|------------|--------|  
| **Accuracy** | 99.38% |  
| **Precision** | 99.69% |  
| **Recall** | 99.07% |  
| **F1 Score** | 99.39% |  
| **ROC AUC** | 99.97% |  

---

## **📊 Visualizations**  
These plots help analyze model performance:  

✅ **Confusion Matrix** – Shows correct vs incorrect classifications  
✅ **Precision-Recall Curve** – Evaluates precision vs recall trade-offs  
✅ **Loss & Accuracy Curves** – Tracks performance over training epochs  
✅ **Predicted Probability Distribution** – Checks confidence levels  

### **Confusion Matrix**  
![Confusion Matrix](confusion_matrix.png)  

### **Precision-Recall Curve**  
![PR Curve](precision_recall_curve.png)  

### **Training Curves**  
![Loss & Accuracy](training_curves.png)  

### **Prediction Distribution**  
![Predicted Probabilities](predictions_distribution.png)  

---

## **📢 Contributing**  
Want to improve the model? Follow these steps:  
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

## **📜 Citation**  
If you use this work, please cite:  
```bibtex
@dataset{zago2020umudga,
  author = {Mattia Zago, Manuel Gil Pérez, Gregorio Martínez Pérez},
  title = {UMUDGA - University of Murcia Domain Generation Algorithm Dataset},
  year = {2020},
  publisher = {Mendeley Data},
  version = {V1},
  doi = {10.17632/y8ph45msv8.1}
}
```

---

## **📜 License**  
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## **🌟 Acknowledgments**  
Special thanks to:  
- **UMUDGA Dataset creators and NetLab 360** for providing **DGA samples**.  
- **Alexa Top 1M** for legitimate domain lists.  
- **TensorFlow, PyTorch, and FastText teams** for open-source contributions.
- **Researchers in AI & Cybersecurity** for inspiration.

---

## **📌 Final Notes**  
This README provides **everything needed** to understand, run, and improve your **DGA detection system**. Would you like to add anything specific? 😊

## **📜 License**
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.


### **📌 Want to Improve DGA Detection?**
🔗 [**Check out the full project on GitHub!**](https://github.com/zenbenali/HSDGA-NLP)  

🚀 **Star this repo if you find it useful!** ⭐

---

### **📌 Abdhine Ben Ali**

