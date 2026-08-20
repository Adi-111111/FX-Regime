import joblib
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

class RegimeEngine:
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=1000,
            random_state=42,
        )
        self.scaler = StandardScaler()

    def fit_regimes(self, latent_features):
        scaled = self.scaler.fit_transform(latent_features)
        self.model.fit(scaled)
        return self.model.predict(scaled)

    def get_regime_probabilities(self, current_data):
        scaled = self.scaler.transform(current_data.reshape(1, -1))
        return self.model.predict_proba(scaled)[0]

    def save_engine(self, path="models/regime_hmm.pkl"):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)

    def load_engine(self, path="models/regime_hmm.pkl"):
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
