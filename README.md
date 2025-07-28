
## 📜 Overview

This project explores the challenge of **detecting fake news** by applying various machine learning and deep learning models to the public **LIAR dataset**. The goal is to classify news statements into binary categories: "Fake" or "Real".

We implement and compare the performance of traditional models like Logistic Regression with more complex deep learning architectures like LSTMs to evaluate their effectiveness in identifying misinformation.

---

## ✨ Key Features

- **Data Preprocessing**: Thoroughly cleans and prepares text data by converting it to lowercase, removing special characters, and filtering out stopwords using NLTK.
- **Feature Extraction**: Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text statements into a meaningful numerical format for machine learning models.
- **Model Comparison**: Trains and evaluates three distinct baseline models:
  - **Logistic Regression**
  - **Multinomial Naive Bayes**
  - **SGD Classifier**
- **Deep Learning Implementation**: Builds a sequential **LSTM (Long Short-Term Memory)** model using TensorFlow/Keras to capture contextual information from the text sequences.

---

## 📊 Model Performance

A comparative analysis of the models was performed on the test set. The accuracy scores are summarized below.

| Model | Accuracy Score |
| :---------------------- | :--------------: |
| **Logistic Regression** |      ~60.0%      |
| **Multinomial Naive Bayes** |      ~59.8%      |
| **SGD Classifier** |      ~59.2%      |
| **LSTM** |      ~55.2%      |

**Observation**: The traditional machine learning models established a strong baseline accuracy of around 60%. The LSTM model showed signs of overfitting during training and did not generalize as well to the validation data, indicating a need for further hyperparameter tuning or architectural adjustments.

---

## 🛠️ Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - **Data Manipulation**: Pandas, NumPy
  - **Text Processing**: NLTK, Scikit-learn (TfidfVectorizer)
  - **Machine Learning**: Scikit-learn
  - **Deep Learning**: TensorFlow, Keras
- **Environment**: Jupyter Notebook

---

## 🚀 How to Run This Project

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/amriteshwarpandey/your-repo-name.git](https://github.com/amriteshwarpandey/your-repo-name.git)
cd your-repo-name
```

### 2. Download the Dataset
Download the LIAR dataset from [this link](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip).
- Unzip the file.
- Place the `train.tsv`, `test.tsv`, and `valid.tsv` files into a directory named `liar_dataset` inside the project folder.

### 3. Set Up a Virtual Environment (Recommended)
```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 4. Install Dependencies
Create a `requirements.txt` file with the following content:
```
pandas
numpy
nltk
scikit-learn
tensorflow
jupyter
```
Then, run the installation command:
```bash
pip install -r requirements.txt
```

### 5. Download NLTK Stopwords
Run this command in a Python interpreter to download the necessary NLTK data:
```python
import nltk
nltk.download('stopwords')
```

### 6. Launch Jupyter Notebook
```bash
jupyter notebook
```
Open the `.ipynb` file and run the cells.

---

## 🔮 Future Improvements

- **Hyperparameter Tuning**: Optimize the LSTM model by adjusting learning rates, dropout values, and the number of layers to combat overfitting.
- **Implement BERT**: Correctly implement and train a `TFBertForSequenceClassification` model from the Hugging Face `transformers` library to leverage a state-of-the-art architecture for NLP tasks.
- **Web Application**: Deploy the best-performing model as a simple web application using **Flask** or **Streamlit** where users can input a news statement and get a prediction.
