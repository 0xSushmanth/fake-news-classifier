# Fake News Classifier

## Description

The Fake News Classifier is a machine learning project that accurately classifies news articles as either "Real News" or "Fake News". The project leverages various techniques, including natural language processing (NLP) and ensemble models, to make predictions based on the textual content of news articles. By identifying fake news, this classifier contributes to combating misinformation and promoting accurate information dissemination.

## Features

- Utilizes different machine learning models and techniques to classify news articles.
- Supports a RESTful API to interact with the classifier programmatically.
- Provides interactive API documentation with Swagger UI and FASTAPI for easy testing and exploration.

## Usage

To use the Fake News Classifier, follow the steps below:

1. Clone the repository:

```bash
git clone https://github.com/0xSushmanth/fake-news-classifier.git

```
2. Install the dependencies:
```bash
pip install -r requirements.txt
```
3. Starting the server locally
```bash
uvicorn main:app --reload
```
## Documentation
API usage and documentation are made possible by FASTAPI and GCP available [here](https://fakenews-0012.et.r.appspot.com/docs)

## Models and Techniques Used
The Fake News Classifier incorporates the following models and techniques:
* Logistic Regression: A linear model used to classify news articles based on various features extracted from the text content.
* Naive Bayes: A probabilistic model used to classify news articles by assuming independence among the features.
* Random Forest: An ensemble model that combines multiple decision trees to improve the accuracy and robustness of the predictions.
* XGBoost: An optimized implementation of gradient boosting that uses a combination of tree-based models and linear models to improve the accuracy and speed of the predictions.
* Text Preprocessing: The input text is preprocessed using tokenization, sequence padding, and hashing techniques to convert it into suitable input features for the models.
Please refer to the project's Jupyter Notebook file (fake-news-classifier.ipynb) for a detailed explanation of the models and techniques used.

## Deployment
This project has been deployed on GCP 
To deploy this project run
Make sure you have installed gcloud SDK  and then run
```bash
  gcloud app deploy app.yaml
```
## Contributing
Contributions are always welcome!
See `contributing.md` for ways to get started.
Please adhere to this project's `code of conduct`.
