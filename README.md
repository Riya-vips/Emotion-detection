Emotion Detection Using NLP
This project is a streamlined pipeline to detect emotions from text using Natural Language Processing (NLP) and machine learning. It includes data ingestion, transformation, model training, and a simple Streamlit app for emotion prediction.

Features
Text Preprocessing: Clean and lemmatize raw text (remove URLs, punctuation, numbers, stopwords, user handles).

Vectorization: TF-IDF vectorizer with unigrams and bigrams.

Label Encoding: Converts emotion labels to numeric form.

Model Training: Trains multiple classifiers (Logistic Regression, Multinomial Naive Bayes, Linear SVC) with hyperparameter tuning using GridSearchCV.

Evaluation: Model performance measured by weighted F1-score, precision, recall, and accuracy.

Prediction UI: Interactive Streamlit app for live text emotion prediction with emoji support.

Project Structure
bash
Copy
Edit

├── notebook/data/            # Raw dataset CSV file (emotion_dataset_raw.csv)
├-------       data_ingestion.py
│   │   ├── data_transformation.py    # Text cleaning, transformation, vectorization
│   │   ├── model_trainer.py           # Model training and evaluation
│           
│   ├── utils.py                       # Save/load utilities and model evaluation
│   ├── application.py                        # Streamlit web app for emotion prediction
├── README.md
Setup Instructions
Clone the repo:

bash
Copy
Edit
git clone https://github.com/Riya-vips/emotion-detection.git
cd emotion-detection
Create and activate virtual environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate      # Linux/macOS
.\venv\Scripts\activate       # Windows
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Download NLTK data (wordnet and omw-1.4):

python
Copy
Edit
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
Run the training pipeline to build and save models:

bash
Copy
Edit
python src/pipeline/train_pipeline.py
Run the Streamlit app:

bash
Copy
Edit
streamlit run src/app.py
Usage
Open the Streamlit app in your browser.

Input any text sentence.

Click Analyze to get the detected emotion label along with an emoji.

Navigate to Info for project details.

Dataset
The dataset used is emotion_dataset_raw.csv which contains text samples labeled with emotions such as anger, joy, sadness, surprise, etc.

Located in notebook/data/



Can be extended to include more advanced models or preprocessing steps.

License
MIT License
