# models/Url_Classifier.py

import re
import math
import pickle
import os
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class URLThreatClassifier:

    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            ))
        ])
        self.is_trained = False

    # ── Feature Engineering ─────────────────────────
    def extract_features(self, url: str) -> list:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().replace('www.', '')
            path   = parsed.path
        except Exception:
            domain, path = '', ''

        features = [
            len(url),                          # 0
            len(domain),                       # 1
            len(path),                         # 2
            url.count('.'),                   # 3
            url.count('-'),                   # 4
            url.count('@'),                   # 5
            url.count('//'),                  # 6
            url.count('?'),                   # 7
            url.count('='),                   # 8
            url.count('%'),                   # 9
            len(domain.split('.')),           # 10
            sum(c.isdigit() for c in domain), # 11
            1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', domain) else 0,  # 12
            1 if url.lower().startswith('https') else 0,           # 13
            1 if any(w in url.lower() for w in [
                'secure','login','verify','bank','account','update',
                'paypal','signin','password','click','free','win'
            ]) else 0,                                            # 14
            self._shannon_entropy(domain)                         # 15
        ]

        # ✅ FIXED
        assert len(features) == 16, f"Expected 16 features, got {len(features)}"
        return features

    # ── Entropy ────────────────────────────────────
    def _shannon_entropy(self, text: str) -> float:
        if not text:
            return 0.0
        freq = {}
        for ch in text:
            freq[ch] = freq.get(ch, 0) + 1
        entropy = 0.0
        for count in freq.values():
            p = count / len(text)
            entropy -= p * math.log2(p)
        return round(entropy, 4)

    # ── Risk Level ─────────────────────────────────
    def _get_risk_level(self, prob: float) -> str:
        if prob >= 0.8:
            return "CRITICAL"
        elif prob >= 0.6:
            return "HIGH"
        elif prob >= 0.4:
            return "MEDIUM"
        return "LOW"

    # ── Train ──────────────────────────────────────
    def train(self, urls: list, labels: list):

        features = [self.extract_features(u) for u in urls]

        if len(urls) < 4:
            self.pipeline.fit(features, labels)
            self.is_trained = True
            print("Model trained (small dataset)")
            return self

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))

        return self

    # ── Predict ────────────────────────────────────
    def predict(self, url: str) -> dict:

        if not self.is_trained:
            raise RuntimeError("Train or load model first")

        extracted = self.extract_features(url)
        features = [extracted]

        proba = self.pipeline.predict_proba(features)[0]
        label = int(self.pipeline.predict(features)[0])

        malicious_prob = float(proba[1]) if len(proba) > 1 else 0.0

        return {
            "prediction": "malicious" if label == 1 else "safe",
            "confidence": round(max(proba) * 100, 2),
            "malicious_score": round(malicious_prob * 100, 2),
            "risk_level": self._get_risk_level(malicious_prob),
            "features": {
                "url_length": extracted[0],
                "domain_length": extracted[1],
                "has_ip": bool(extracted[12]),
                "has_https": bool(extracted[13]),
                "suspicious_keywords": bool(extracted[14]),
                "entropy": extracted[15]
            }
        }

    # ── Save ───────────────────────────────────────
    def save(self, path="models/url_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)

    # ── Load ───────────────────────────────────────
    def load(self, path="models/url_model.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError("Train model first")
        with open(path, "rb") as f:
            self.pipeline = pickle.load(f)
        self.is_trained = True
        return self