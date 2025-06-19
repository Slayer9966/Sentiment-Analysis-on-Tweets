# 💬 Twitter Keyword-Based Sentiment Analysis using BERT + BiLSTM

Analyze real-time Twitter sentiments using BERT embeddings and a Bidirectional LSTM model. This project fetches tweets and their replies based on a user-defined keyword and classifies the overall sentiment as **positive**, **negative**, or **neutral**.

---

## 🚀 Project Overview

Social media is full of opinions, and understanding public sentiment behind those opinions can provide critical insights for businesses, brands, and researchers. This project performs **keyword-based sentiment analysis** on tweets using a combination of:

- **Tweet Scraping** via Twitter API v2  
- **Text Vectorization** using **BERT**
- **Sentiment Classification** using a **BiLSTM** model

---

## 🧠 How It Works

1. **Keyword-Based Tweet Scraping**
   - Scrapes 10 tweets containing the keyword using the Twitter API.
   - Fetches replies for those tweets to provide context.

2. **Text Preprocessing**
   - Cleans tweet and reply texts (removes URLs, stopwords, special characters).
   - Tokenizes and pads the text for model compatibility.

3. **Feature Extraction with BERT**
   - Text is converted into high-dimensional embeddings using **BERT**.
   - These embeddings are saved as `bert_embeddings.pkl`.

4. **Sentiment Classification with BiLSTM**
   - A **BiLSTM model** is trained on the BERT embeddings.
   - Trained model is saved as `best_bilstm_model.keras`.

5. **Sentiment Prediction**
   - The model classifies each tweet and its replies into one of three categories:
     - Positive
     - Neutral
     - Negative

---

## 🗂️ Project Structure

```bash
📁 Twitter-Sentiment-Analysis/
├── vectorization.ipynb               # Preprocessing and BERT embedding
├── Model_Training_BLSTM_KERAS.ipynb  # BiLSTM model training
├── Twitter_Sentiment.ipynb           # Real-time sentiment analysis
├── bert_embeddings.pkl               # Saved BERT embeddings
├── best_bilstm_model.keras           # Trained BiLSTM model
└── README.md                         # Project documentation
```

---
## 🗃️ Dataset

This project was trained using the **[Twitter Sentiment Analysis Dataset (Sentiment140)](https://www.kaggle.com/datasets/kazanova/sentiment140)** from Kaggle. It contains **1.6 million labeled tweets**, providing a balanced dataset for sentiment classification into **positive**, **neutral**, and **negative** categories.

---

## 🛠️ Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/Twitter-Sentiment-Analysis.git
cd Twitter-Sentiment-Analysis
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```



3. **Vectorize the Data with BERT**

Run `vectorization.ipynb` to:
- Preprocess your tweets.
- Generate BERT embeddings.
- Save them as `bert_embeddings.pkl`.

4. **Train the BiLSTM Model**

Run `Model_Training_BLSTM_KERAS.ipynb`:
- Loads BERT embeddings.
- Trains a BiLSTM classifier.
- Saves the model as `best_bilstm_model.keras`.

> You may skip this step and directly use the provided model and embeddings if available.

5. **Run Real-Time Sentiment Analysis**

Open `Twitter_Sentiment.ipynb`:
- Set your **keyword** and **Twitter Bearer Token**:
```python
keyword = "AI technology"
BEARER_TOKEN = "your_twitter_api_token"
```
- Load model and BERT embeddings:
```python
model = keras.models.load_model("best_bilstm_model.keras", compile=False)

with open("bert_embeddings.pkl", "rb") as f:
    # Load your BERT vectorizer
```
- Run the notebook to see predictions for the tweets and their replies.

---

## 📝 Notes

- Due to API rate limits, the script fetches only **10 tweets and their replies** by default.
- You can modify the logic to:
  - First get 10 tweets containing the **keyword**.
  - Then fetch replies to those tweets for deeper sentiment analysis.
- Ensure you have access to the [Twitter Developer Portal](https://developer.twitter.com/) to generate your Bearer Token.

---

## 📊 Results

- Achieved **80% accuracy** on a balanced dataset using BERT + BiLSTM.
- Successfully predicted real-time sentiment from public Twitter data based on any keyword.

---

## 📌 Future Improvements

- Add support for more replies per tweet via pagination.
- Visualize sentiment trends across time or geography.
- Integrate a Flask API or Streamlit UI for live keyword analysis.

---

## 📃 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Twitter Developer API](https://developer.twitter.com/)
- [TensorFlow](https://www.tensorflow.org/)

---

> ## 🙋‍♂️ Author

**Syed Muhammad Faizan Ali**  
📍 Islamabad, Pakistan  
📧 faizandev666@gmail.com  
🔗 [GitHub](https://github.com/Slayer9966) | [LinkedIn](https://www.linkedin.com/in/faizan-ali-7b4275297/)
📢 If you find this project helpful or use it in your work, please consider giving it a ⭐ or letting me know via email or GitHub issues!
