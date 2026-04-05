# SMS Spam Classification using NLP

## Project Overview

This project implements a **Text Classification system** to automatically classify SMS messages as **Ham (legitimate)** or **Spam** using Natural Language Processing (NLP) techniques and Machine Learning. The model uses **Multinomial Naive Bayes** classifier with **TF-IDF vectorization** to achieve high accuracy in spam detection.

---

## Dataset Details

### Dataset Statistics
- **Total Messages**: 5,572
- **Ham (Legitimate)**: 4,825 messages (86.6%)
- **Spam**: 747 messages (13.4%)
- **Format**: CSV file with two columns
  - `v1`: Label (ham or spam)
  - `v2`: Raw SMS text message

### Dataset Structure
Each line in the dataset contains one message with two columns:
- **Column 1 (v1)**: Label indicating whether the message is "ham" (legitimate) or "spam"
- **Column 2 (v2)**: The raw text of the SMS message

### Sample Dataset Rows

| Label | Message |
|-------|---------|
| ham | "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..." |
| ham | "Ok lar... Joking wif u oni..." |
| spam | "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's" |
| ham | "U dun say so early hor... U c already then say..." |
| ham | "Nah I don't think he goes to usf, he lives around here though" |

---

## Dataset Sources & Corpus Information

This SMS Spam Collection dataset has been compiled from multiple free or research-available sources:

### 1. **Grumbletext Web Forum SMS Spam**
- **Size**: 425 SMS spam messages
- **Source**: Grumbletext Web site - A UK forum where cell phone users report SMS spam
- **Collection Method**: Manually extracted from public complaints
- **Challenge**: Manual identification of spam messages from user claims was a time-consuming task requiring careful scanning of hundreds of web pages

### 2. **NUS SMS Corpus (NSC) - Ham Messages**
- **Size**: 3,375 SMS randomly selected legitimate messages
- **Source**: National University of Singapore, Department of Computer Science
- **Total NUS Corpus Size**: ~10,000 legitimate messages
- **Origin**: Primarily from Singaporeans and University students
- **Collection Method**: Volunteers were informed their contributions would be made publicly available
- **Availability**: Public dataset available for research

### 3. **Caroline Tag's PhD Thesis SMS Collection**
- **Size**: 450 SMS ham (legitimate) messages
- **Source**: Academic research collection from PhD thesis
- **Availability**: Publicly available academic resource

### 4. **SMS Spam Corpus v.0.1 Big**
- **Ham Messages**: 1,002 SMS
- **Spam Messages**: 322 SMS
- **Total**: 1,324 SMS messages
- **Availability**: Publicly available corpus
- **Academic Use**: This corpus has been utilized in multiple academic research projects

---

## Project Workflow

### Step 1: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 2: Data Loading and Cleaning
- Load dataset from `spam.csv`
- Remove unnecessary columns with missing values
- Rename columns to 'label' and 'text' for clarity

### Step 3: NLP Preprocessing
The following preprocessing techniques are applied to normalize and clean the text data:

1. **Lowercase Conversion**: Convert all text to lowercase
2. **Punctuation Removal**: Remove special characters and punctuation
3. **Tokenization**: Break text into individual words
4. **Stopword Removal**: Eliminate common English words (e.g., "the", "a", "is")
5. **Stemming**: Reduce words to root form using Porter Stemmer (e.g., "running" → "run")
6. **Lemmatization**: Convert words to dictionary form (e.g., "walked" → "walk")

### Step 4: Feature Extraction
- **TF-IDF Vectorization**: Convert processed text into numerical features
- **Maximum Features**: 5,000 terms
- **Output Shape**: (5572, 5000)

### Step 5: Train-Test Split
- **Training Data**: 70% (3,900 messages)
- **Testing Data**: 30% (1,672 messages)
- **Random State**: 42 (for reproducibility)

### Step 6: Model Training
- **Algorithm**: Multinomial Naive Bayes
- **Training**: Fitted on training data with TF-IDF features

### Step 7: Model Evaluation

#### Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 96.76% |
| **Precision (Spam)** | 100% |
| **Recall (Spam)** | 74.88% |
| **F1-Score (Spam)** | 85.89% |

#### Detailed Classification Report

```
              precision    recall  f1-score   support

           0       0.96      0.98      0.97      1443
           1       1.00      0.75      0.86       229

    accuracy                           0.97      1672
   macro avg       0.98      0.86      0.91      1672
weighted avg       0.97      0.97      0.97      1672
```

#### Confusion Matrix

```
Predicted:
           Ham    Spam
Actual Ham  1413   30
Actual Spam 58    171
```

**Interpretation**:
- **True Negatives (TN)**: 1,413 (correctly classified as ham)
- **True Positives (TP)**: 171 (correctly classified as spam)
- **False Negatives (FN)**: 58 (spam classified as ham)
- **False Positives (FP)**: 30 (ham classified as spam)

---

## Key Findings

1. **High Accuracy**: The model achieves 96.76% accuracy, demonstrating excellent overall performance
2. **Perfect Precision**: 100% precision for spam detection means no legitimate messages are incorrectly flagged as spam
3. **Good Recall**: 74.88% recall indicates the model catches approximately 75% of actual spam messages
4. **Balanced Performance**: The model shows good balance between sensitivity and specificity

---

## How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/Aryankr0711/Text_classification-.git
cd Text_classification-
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook Aryan_MDM_NLP.ipynb
```

Or use Jupyter Lab:
```bash
jupyter lab Aryan_MDM_NLP.ipynb
```

### 4. Execute All Cells
- Run all cells in the notebook to execute the complete pipeline
- View the model performance metrics and visualizations

---

## Technologies Used

- **Python 3.x**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **NLTK** (Natural Language Toolkit): NLP preprocessing
  - Tokenization
  - Stopword removal
  - Stemming and Lemmatization
- **Scikit-learn**: Machine learning algorithms and evaluation metrics
  - TF-IDF Vectorization
  - Multinomial Naive Bayes
  - Train-test split
  - Classification metrics

---

## Project Structure

```
Text_classification-/
├── Aryan_MDM_NLP.ipynb          # Main Jupyter notebook with complete pipeline
├── spam.csv                      # Dataset file (5,572 SMS messages)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Future Improvements

1. **Try Advanced Models**: Implement SVM, Random Forest, or Deep Learning models (LSTM/CNN)
2. **Hyperparameter Tuning**: Optimize TF-IDF and model parameters
3. **Class Balancing**: Address the class imbalance (ham: 86.6%, spam: 13.4%)
4. **Cross-Validation**: Implement k-fold cross-validation for robust evaluation
5. **Feature Engineering**: Extract domain-specific features (phone numbers, URLs, etc.)
6. **Ensemble Methods**: Combine multiple models for better predictions

---

## Author

**Aryan Kumar** & **Siddhant Sahu**

---

## License

This dataset and project are available for educational and research purposes. Please refer to the original dataset sources for specific licensing information.

---

## References

- **NLTK Documentation**: https://www.nltk.org/
- **Scikit-learn Documentation**: https://scikit-learn.org/
- **NUS SMS Corpus**: National University of Singapore
- **SMS Spam Corpus**: Publicly available academic resource
- **Grumbletext Forum**: SMS spam complaints repository

---

## Contact & Support

For questions or suggestions about this project, please open an issue or contact the project authors (Aryan Kumar & Siddhant Sahu).

