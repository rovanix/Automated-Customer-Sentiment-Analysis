# Automated-Customer-Sentiment-Analysis
# NLP Pipeline: Automated Customer Sentiment Analysis

**Student Name**: Somesh Ranjan Rout  
**Student ID**: GH1039569  
**Course** : M508C Big Data Analytics  
**Date**: 18/12/2025  
**Dataset**: [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

# 1. Problem Statement: Automated Audience Sentiment Analysis

### 1.1 The Business Problem
In the modern entertainment industry, the volume of user-generated content (reviews, tweets, comments) is too vast for manual monitoring. Our company needs to understand audience sentiment toward media content to inform marketing strategies, greenlight future projects, and manage brand reputation. Manually reading thousands of reviews is inefficient and prone to human bias.

### 1.2 Importance and Benefit
Automating sentiment analysis allows the company to:
* **Rapidly gauge public reaction** to new releases.
* **Identify specific pain points** or highlights mentioned by the audience.
* **Reduce operational costs** by automating a task that previously required a dedicated PR team.

### 1.3 Data Collection and Task Formulation
* **Data Collection:** The system utilizes the **IMDB Dataset**, consisting of 50,000 highly polar movie reviews pulled from public repositories and cleaned of HTML artifacts.
* **NLP Task Formulation:** This is formulated as a **Binary Sentiment Classification** task. Given a text string (review), the system must predict a label of `1` (Positive) or `0` (Negative).

# 2. High-Level System Design

Our NLP pipeline is designed as a modular flow to ensure scalability and ease of testing. Each component is necessary to transform raw, noisy human language into structured data that a machine can learn from.

### 2.1 Main Components and Connectivity
1. **Text Preprocessing:** Raw text is cleaned of HTML tags, URLs, and emojis. It is then tokenized, stripped of stopwords, and lemmatized to reduce vocabulary complexity.
2. **Feature Engineering (Vectorization):** The cleaned tokens are converted into numerical vectors. We use **TF-IDF** for statistical importance and **Word2Vec** or **Tokenization sequences** for neural architectures.
3. **Modeling Engine:** The system feeds these vectors into multiple models, ranging from **Logistic Regression** (baseline) to **LSTM** (sequential dependencies) and **BERT** (contextual embeddings).
4. **Evaluation Suite:** Predictions are compared against ground-truth labels using a suite of metrics to ensure model reliability before business deployment.

# 3. Detailed Design and Implementation

### 3.1 Data Preparation Strategy
To ensure the model learns meaningful patterns rather than noise, we implement a rigorous cleaning pipeline:
* **Cleaning:** BeautifulSoup removes HTML; regex handles special characters.
* **Normalization:** Lemmatization groups different forms of a word (e.g., "running" to "run") to consolidate the feature space.

### 3.2 Modeling Approach
We implement a multi-tiered modeling strategy to find the optimal balance between accuracy and computational cost:
* **Statistical Model:** Logistic Regression with TF-IDF to capture keyword-based sentiment.
* **Sequence Model:** Long Short-Term Memory (LSTM) networks to capture the order of words in a review.
* **Transformer Model:** BERT/DistilBERT to leverage pre-trained linguistic knowledge for high-accuracy contextual understanding.

## Importing Libraries

```python
# BASIC LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Libraries for downloading data
import os
import urllib.request

# TEXT PROCESSING
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
# POS & NER (spaCy)
import spacy
# LANGUAGE DETECTION
!pip install langdetect
from langdetect import detect
# SENTIMENT ANALYSIS
from textblob import TextBlob
# READABILITY SCORES
!pip install textstat
import textstat
# FEATURE ENGINEERING: TF-IDF & N-GRAMS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# WORD2VEC (FOR LSTM)
!pip install gensim
from gensim.models import Word2Vec, KeyedVectors
# TRAIN / TEST SPLIT & IMPBALANCED HANDLING
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
# SMOTE for imbalance
from imblearn.over_sampling import SMOTE
# MACHINE LEARNING MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# LSTM (DEEP LEARNING)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# TRANSFORMER MODELS (BERT, DISTILBERT)
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from transformers import AdamWeightDecay
# LEVENSHTEIN DISTANCE (Edit distance V1 & V2)
!pip install python-Levenshtein
import Levenshtein
# MODEL EVALUATION
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)
# VISUALIZATION (WORD CLOUD)
from wordcloud import WordCloud
# WARNINGS
import warnings
warnings.filterwarnings("ignore")

print("All libraries imported successfully!")
