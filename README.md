# CyberGuard-AI


![image](https://github.com/user-attachments/assets/49ce89d6-cb41-4752-bc28-90a44577983b)
This project is a machine learning solution designed for the **IndiaAI CyberGuard AI Hackathon**. The challenge is to develop a machine learning model that analyzes and classifies unstructured text data related to cybercrimes to improve citizen support on the **National Cyber Crime Reporting Portal (NCRP)**. This repository contains the implementation of the solution using Natural Language Processing (NLP) and machine learning techniques, including the use of **BERT (Bidirectional Encoder Representations from Transformers)**.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Setup and Installation](#setup-and-installation)
4. [Preprocessing](#preprocessing)
5. [Model Selection and Training](#model-selection-and-training)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Visualizations](#visualizations)
8. [Results](#results)
9. [Future Work](#future-work)

## Project Overview

This project focuses on automating the classification of cybercrime reports submitted by citizens to the **National Cyber Crime Reporting Portal (NCRP)**. The goal is to build a machine learning model capable of categorizing and sub-categorizing reports into predefined classes, which include various types of cybercrime like fraud, identity theft, and cyberbullying.

The aim is to improve the speed and accuracy of processing cybercrime reports, helping law enforcement and authorities handle these reports in a timely manner. The model also aims to reduce the burden on human agents who would otherwise have to manually categorize incoming cybercrime reports.

### Key Features:
- Text preprocessing for cleaning and preparing raw data.
- Classification using **BERT**, a powerful NLP model, for accurate categorization.
- Evaluation using common metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
- Visualizations to better understand model performance, including precision-recall bar plots and confusion matrices.

## Dataset

The dataset used consists of cybercrime reports from citizens submitted to the **National Cyber Crime Reporting Portal (NCRP)**. The dataset contains both training and test data and includes fields such as crime descriptions, categories, subcategories, and additional metadata for classification.

- **Categories**: These represent broader crime types (e.g., Financial Fraud, Cyber Terrorism, Online Bullying).
- **Subcategories**: These represent more specific types of crimes within the broader categories (e.g., Debit/Credit Card Fraud, Internet Banking Fraud).
- **Text Descriptions**: Unstructured text submitted by citizens detailing the crime incident.

The dataset was preprocessed to handle missing values, clean the text, and prepare it for tokenization and classification.

## Setup and Installation

Follow these steps to set up the environment and run the project locally.



### Requirements

- Python 3.7+
- Install the required libraries by running:

```bash
pip install -r requirements.txt
git clone https://github.com/yourusername/CyberGuard-AI-Hackathon.git
cd CyberGuard-AI-Hackathon
jupyter notebook CyberGuard_AI_Hackathon.ipynb

//Model training
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
```
### Preprocessing

Data preprocessing is a critical step in preparing the text data for machine learning models. The following steps were implemented:

Filling Missing Values: Missing sub_category values were filled with the placeholder 'Unknown'.
Text Cleaning: This involves several steps to standardize the text data:
Lowercasing: All text was converted to lowercase to ensure uniformity.
Removing Special Characters and Numbers: Non-alphabetic characters such as punctuation marks and numbers were removed, as they do not contribute meaningfully to the classification task.
Whitespace Handling: Extra spaces and newline characters were removed to clean up the data further.
Tokenization: The text data was tokenized into individual words or tokens. This is a necessary step for transforming the text into a format that can be fed into machine learning models.
Label Encoding: Both categories and subcategories were encoded as integers using LabelEncoder to prepare the data for classification.


### Model Selection and Training

Several models were considered for the classification task:

Logistic Regression
Random Forest
LSTM (Long Short-Term Memory)
BERT (Bidirectional Encoder Representations from Transformers)
Why BERT?
BERT is a transformer-based architecture that uses attention mechanisms to understand the context of words within sentences. Unlike traditional models that process words sequentially, BERT processes the entire sentence in parallel, enabling it to understand bidirectional context. This makes BERT particularly well-suited for natural language understanding tasks like text classification.

### Model Training Process:
Splitting Data: The data was split into training and validation sets using stratified sampling to maintain class distribution in both sets.
Training: The BERT model was fine-tuned using the training data for three epochs. The optimizer used was AdamW, which is commonly used for transformer models.
Evaluation: After training, the model was evaluated using the validation data and performance metrics such as accuracy, precision, recall, F1-score, and the confusion matrix.
Model Training Code Example:


from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

### Confusion Matrix
The confusion matrix provides a detailed breakdown of predictions across all classes. It helps visualize how well the model is distinguishing between different classes. The diagonal elements represent the correctly classified instances, while off-diagonal elements represent misclassifications.

Example Classification Report:
plaintext

                     precision    recall  f1-score   support
       Online Financial Fraud   0.92      0.95      0.93      11470
   Child Pornography/CSAM      0.85      0.80      0.82         84
   Cyber Terrorism              0.50      0.40      0.44         38
   Any Other Cyber Crime        0.76      0.72      0.74      2142


### Results
The final model achieves an overall accuracy of 89% across all classes. The confusion matrix and classification report indicate that the model performs well for categories like "Online Financial Fraud," but faces challenges with underrepresented categories like "Cyber Terrorism."

![image](https://github.com/user-attachments/assets/843d8c71-43f8-4209-9677-3acdb2208b65)

Key Metrics:
Accuracy: 89%
Macro Avg. Precision: 75%
Macro Avg. Recall: 69%
Macro Avg. F1-Score: 71%
The model’s high precision in certain classes (e.g., Financial Fraud) suggests it is effective at predicting these classes, while low recall in others (e.g., Cyber Terrorism) indicates that the model might need further improvement, especially in handling rare categories.

### Future Work
Future improvements could include:

Class Balancing: Implementing techniques like oversampling or SMOTE to handle class imbalance.
Model Tuning: Experimenting with different hyperparameters such as learning rate and batch size to improve model performance.
Advanced Models: Moving beyond BERT to other transformer models like GPT, which may offer even better performance in text classification tasks.
Real-Time Integration: Deploying the model as an API to allow real-time classification of new cybercrime reports on NCRP.


