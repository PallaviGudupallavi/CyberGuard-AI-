# CyberGuard-AI

![image](https://github.com/user-attachments/assets/49ce89d6-cb41-4752-bc28-90a44577983b)
![image](https://github.com/user-attachments/assets/843d8c71-43f8-4209-9677-3acdb2208b65)

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
10. [License](#license)

## Project Overview

This project focuses on automating the classification of cybercrime reports submitted by citizens to the **National Cyber Crime Reporting Portal (NCRP)**. The goal is to build a machine learning model capable of categorizing and sub-categorizing reports into predefined classes, which include various types of cybercrime like fraud, identity theft, and cyberbullying.

### Key Features:
- Text preprocessing for cleaning and preparing raw data.
- Classification using **BERT**, a powerful NLP model, for accurate categorization.
- Evaluation using common metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
- Visualizations to better understand model performance, including precision-recall bar plots and confusion matrices.

## Dataset

The dataset consists of unstructured text descriptions of cybercrimes, along with associated category and subcategory labels. The data was provided for the **IndiaAI CyberGuard AI Hackathon**.

- **Categories**: E.g., Financial Fraud, Child Sexual Abuse Material (CSAM), Cyber Terrorism, etc.
- **Subcategories**: Specific crimes under each category (e.g., Internet Banking Fraud, SIM Swap Fraud).

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

---

### How to Use:
1. **Replace** `yourusername` with your GitHub username where the images are hosted.
2. **Upload** the `README.md` file to your GitHub repository.
3. **Ensure** that the images are stored in the correct location in your GitHub repository (as referenced in the markdown).

This `README.md` is now ready for your GitHub repository! Let me know if you need further adjustments.
