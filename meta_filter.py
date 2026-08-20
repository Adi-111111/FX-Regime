import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

class MetaFilter:
    def __init__(self, n_estimators=200, max_depth=5):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight='balanced_subsample',
            n_jobs=-1,
        )
        self.threshold = 0.4

    def train_model(self, X, y):
        self.model.fit(X, y)
        probs = self.model.predict_proba(X)[:, 1]

        best_prec, best_thresh = 0, 0.5
        for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            preds = (probs >= t).astype(int)
            count = np.sum(preds)
            if count > 0:
                prec = precision_score(y, preds)
                if prec > best_prec and count > 10:
                    best_prec, best_thresh = prec, t

        self.threshold = best_thresh
        print(f"threshold set to {self.threshold:.2f}, precision {best_prec:.2f}")

    def get_veto_decision(self, latent_features):
        if latent_features.ndim == 1:
            latent_features = latent_features.reshape(1, -1)
        prob = self.model.predict_proba(latent_features)[0, 1]
        return (prob >= self.threshold), prob

    def get_position_sizing(self, probability):
        if probability < self.threshold:
            return 0.0
        edge = probability - self.threshold
        multiplier = 1.0 + (edge * 10.0)
        return min(max(multiplier, 0.5), 2.0)

    def get_feature_importance(self, feature_names=None):
        importances = self.model.feature_importances_
        if feature_names:
            return sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        return importances

    def save_model(self, path='models/meta_filter.pkl'):
        joblib.dump(self, path)

    def load_model(self, path='models/meta_filter.pkl'):
        loaded = joblib.load(path)
        self.model = loaded.model
        self.threshold = loaded.threshold
