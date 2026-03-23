# email_classifier.py
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class EmailPhishingClassifier:
    """
    Classifies emails as 'phishing' or 'legitimate'.
    
    HOW IT WORKS (say this to recruiter):
    1. TF-IDF converts email text into numbers (a vector of word importance scores)
    2. Naive Bayes learns which words are statistically likely in phishing vs legit emails
    3. On a new email, it calculates probability of each class and picks the higher one
    """
    
    def __init__(self):
        # Pipeline chains two steps: vectorizer -> classifier
        # This is a best practice — it prevents data leakage during cross-validation
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,   # Only use top 10k most important words
                ngram_range=(1, 2),   # Use single words AND two-word phrases
                stop_words='english', # Remove "the", "is", "at" etc.
                lowercase=True
            )),
            ('clf', MultinomialNB(alpha=1.0))  # alpha = Laplace smoothing
        ])
        self.is_trained = False
    
    def train(self, texts: list[str], labels: list[int]):
        """
        texts: list of email bodies
        labels: 0 = legitimate, 1 = phishing
        """
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate performance
        y_pred = self.pipeline.predict(X_test)
        print(classification_report(y_test, y_pred, 
              target_names=['Legitimate', 'Phishing']))
        
        return self
    
    def predict(self, email_text: str) -> dict:
        """Returns prediction with confidence score"""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
        
        # predict_proba returns [prob_legit, prob_phishing]
        proba = self.pipeline.predict_proba([email_text])[0]
        label = self.pipeline.predict([email_text])[0]
        
        return {
            "prediction": "phishing" if label == 1 else "legitimate",
            "confidence": round(float(max(proba)) * 100, 2),
            "phishing_score": round(float(proba[1]) * 100, 2),
            "risk_level": self._get_risk_level(proba[1])
        }
    
    def _get_risk_level(self, phishing_prob: float) -> str:
        if phishing_prob >= 0.8:
            return "CRITICAL"
        elif phishing_prob >= 0.6:
            return "HIGH"
        elif phishing_prob >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def save(self, path: str = "models/email_model.pkl"):
        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f)
    
    def load(self, path: str = "models/email_model.pkl"):
        with open(path, 'rb') as f:
            self.pipeline = pickle.load(f)
        self.is_trained = True
        return self