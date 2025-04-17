# 🎭 Vector-Space-Semantics-for-Similarity-between-Friends-Characters


---

## 📌 Project Overview

This project demonstrates a complete NLP pipeline applied to character dialogue from a television script dataset. The objective was to **analyze, represent, and compare dialogue styles across characters** using advanced text processing and machine learning techniques.

The workflow involved:
- Data preprocessing (cleaning, tokenization, lemmatization, etc.)
- Rich linguistic feature extraction (n-grams, POS tagging, sentiment, etc.)
- Context-aware document creation per character
- Feature transformation (TF-IDF, chi-squared feature selection)
- Cosine similarity analysis and model evaluation

---

## 📂 Dataset Summary

- Script-based dialogue data from multiple characters
- Split into training and held-out test datasets
- Total characters: **10**
- Total words: **68,416** in training set, **8,598** in test set

### 📈 Lines per Character (Train Example)
| Character         | Lines | Words |
|------------------|-------|-------|
| Chandler Bing     | 355   | 7,944 |
| Rachel Green      | 355   | 8,359 |
| Monica Geller     | 305   | 6,824 |
| Joey Tribbiani    | 315   | 7,389 |
| Ross Geller       | 315   | 6,923 |
| Phoebe Buffay     | 300   | 6,170 |
| Other_Male        | 305   | 5,808 |
| Other_Female      | 340   | 6,877 |
| Other_None        | 320   | 5,271 |
| #ALL#             | 300   | 6,851 |

---

## ⚙️ Key Functions and Features

### 1. **Preprocessing**
- Lowercasing, whitespace normalization
- Removal of stopwords, punctuation, and numbers
- Stemming and lemmatization for normalization

### 2. **Linguistic Feature Extraction**
- Unigrams to 4-grams
- POS tagging
- Lexical sentiment (polarity and subjectivity)
- Gender prediction via name heuristics

### 3. **Vectorization**
- Dictionary-based feature mapping
- TF-IDF transformation for word importance
- Optional chi-squared feature selection (dimensionality reduction)

### 4. **Contextual Document Creation**
- Per-character document generation from scene-based scripts
- Includes speaker’s lines plus dialogue context (from others)
- Controlled by `context_size` and `max_lines_per_character`

### 5. **Parameter Search**
- Grid search over preprocessing and feature options
- Applied independently on train and validation corpora
- Evaluated using accuracy and rank metrics

### 6. **Similarity Analysis**
- Cosine similarity matrix generation
- Character-wise heatmap visualization
- Evaluation of representational consistency across datasets

---

## 🧪 Final Evaluation

- **Mean Rank:** 1.0  
- **Cosine Similarity:** 1.0  
- **Accuracy:** 10/10 (100%)  
- **Observation:** High accuracy suggests strong model fit, though there is a potential risk of overfitting or overly simple test structure.

---

## 🔍 Key Takeaways

- Advanced linguistic and stylistic features boost classification and profiling accuracy.
- Contextual scene modeling enhances representational depth of each character’s dialogue.
- Diversity in dialogue patterns (e.g., “Other_None”) can affect model confidence.
- Feature selection and parameter tuning play a vital role in generalization.

---

## 🛠️ Technologies Used

- Python 3  
- `nltk`, `sklearn`, `pandas`, `matplotlib`  
- TF-IDF, POS tagging, Chi-squared feature selection  
- Cosine similarity, heatmaps for analysis

---

## 🚀 How to Run

1. Install dependencies:  
   ```bash
   pip install -r requirements.txt
