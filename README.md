
# 🧠 MedAbstract - Medical Abstract Segmentation with DeBERTa v3

Segment and classify sentences from medical research abstracts to improve readability and enable skimmable abstracts using state-of-the-art NLP techniques.

![SkimLit Banner](https://i.postimg.cc/Y0gXWHN9/Abstract-Segmented-Abstract.png)

## 🎯 Objective

Medical abstracts are often dense, complex, and presented as a single block of text—making them hard to skim quickly. This project aims to enhance their readability by **segmenting abstracts line by line into standardized sections** such as `OBJECTIVE`, `METHODS`, `RESULTS`, etc., using a transformer-based NLP model.

We leverage the **DeBERTa-v3 Base** model from HuggingFace, fine-tuned using PyTorch, to classify each line in an abstract into one of several pre-defined categories.

---

## 📁 Dataset

The dataset used is from the paper:  
**"PubMed 20k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts"** (October 2017)

- **Source**: PubMed RCT (Randomized Controlled Trials)
- **Content**: Over 20,000 abstracts from medical literature
- **Task**: Classify each sentence into:
  - `OBJECTIVE`
  - `BACKGROUND`
  - `METHODS`
  - `RESULTS`
  - `CONCLUSIONS`

---

## 🧰 Technologies Used

- **Python**, **Pandas**, **NumPy**
- **PyTorch**, **Transformers (HuggingFace)**
- **DeBERTa v3 Base** model
- **Matplotlib**, **Seaborn**
- **scikit-learn** for evaluation metrics
- **Torchinfo**, **Torchview** for model inspection

---

## 🧪 Key Features

- **Data Preprocessing**:
  - Load and parse `.txt` files
  - Create custom dataset and dataloader for PyTorch

- **Model Architecture**:
  - DeBERTa v3 + Classification head
  - Token classification using HuggingFace `AutoTokenizer`

- **Training Pipeline**:
  - Custom `Dataset` class
  - Training loop with `optimizer`, `scheduler`, and `loss`
  - Evaluation with metrics like `accuracy`, `F1-score`, `Matthews Corrcoef`

- **Visualization**:
  - Loss and accuracy plots
  - Confusion matrix display

---

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers datasets
pip install seaborn matplotlib scikit-learn torchinfo
```

### 2. Download Dataset

The original dataset can be downloaded from [PubMed RCT Dataset on GitHub](https://github.com/Franck-Dernoncourt/pubmed-rct).

Place the extracted files into a `data/` directory.

### 3. Run the Notebook

```bash
jupyter notebook skimlit-debertav3.ipynb
```

---

## 📊 Evaluation Metrics

- **Accuracy**
- **F1-Score**
- **Confusion Matrix**
- **Matthews Correlation Coefficient**

---

## 📌 Results

The fine-tuned DeBERTa model was able to effectively segment and classify sentences from medical abstracts with high accuracy and performance on all major metrics. This demonstrates the potential of transformer-based models in structured text understanding in biomedical NLP.

---

## 📚 References

- [PubMed RCT Dataset (GitHub)](https://github.com/Franck-Dernoncourt/pubmed-rct)
- [DeBERTa v3 (HuggingFace)](https://huggingface.co/microsoft/deberta-v3-base)
- [Original SkimLit Project (Daniel Bourke)](https://github.com/mrdbourke/pytorch-deep-learning)

---

## 🙌 Acknowledgments

Thanks to:
- Microsoft for releasing DeBERTa
- HuggingFace for the `transformers` library
- The authors of the PubMed RCT paper

---

## 📌 Future Improvements

- Add support for attention visualization
- Extend classification to full abstracts with context
- Deploy as an API or streamlit app for public demos
