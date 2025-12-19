from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score

class ModelEvaluator:
    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.models = {
            "Isolation Forest": IsolationForest(contamination=self.contamination, random_state=42),
            "One-Class SVM": OneClassSVM(nu=self.contamination, kernel="rbf", gamma=0.01),
            "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=self.contamination)
        }

    def compare_models(self, X, y_true):
        """
        Trains multiple models and selects the best one based on F1 Score.
        """
        results = {}
        best_model_name = None
        best_score = 0
        
        print(f"Model Comparison (Contamination: {self.contamination})")
        
        for name, model in self.models.items():
            
            if name == "Local Outlier Factor":
                preds = model.fit_predict(X)
            else:
                model.fit(X)
                preds = model.predict(X)
            
            # Convert -1 (Anomaly) / 1 (Normal) to 1 (Anomaly) / 0 (Normal)
            binary_preds = [1 if x == -1 else 0 for x in preds]
            
            # Calculate F1 Score (Harmonic mean of precision and recall)
            score = f1_score(y_true, binary_preds)
            results[name] = score
            
            print(f"[{name}] F1 Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model_name = name

        print(f"\n BEST MODEL: {best_model_name} (F1: {best_score:.4f})")
        return best_model_name, results