import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from pathlib import Path
import re
from typing import List, Dict, Union, Tuple
import logging
from dataclasses import dataclass
import pickle

@dataclass
class SensitivityConfig:
    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 3)
    min_df: int = 2
    n_estimators: int = 100
    max_depth: int = None
    min_samples_split: int = 5
    class_weight: str = 'balanced'
    cache_dir: str = "sensitivity_model_cache"
    model_filename: str = "sensitivity_model.joblib"
    vectorizer_filename: str = "tfidf_vectorizer.pickle"
    custom_patterns: Dict[str, float] = None

class DocumentSensitivityClassifier:
    def __init__(self, config: SensitivityConfig = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or SensitivityConfig()
        self.vectorizer = None
        self.model = None
        self._setup_cache_dir()
        
    def _setup_cache_dir(self):
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        
    def _get_cache_paths(self) -> Tuple[Path, Path]:
        model_path = Path(self.config.cache_dir) / self.config.model_filename
        vectorizer_path = Path(self.config.cache_dir) / self.config.vectorizer_filename
        return model_path, vectorizer_path
    
    def _initialize_vectorizer(self):
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            stop_words='english',
            strip_accents='unicode',
            lowercase=True
        )
    
    def _initialize_model(self):
        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            class_weight=self.config.class_weight,
            n_jobs=-1,
            random_state=42
        )
    
    def _apply_custom_patterns(self, text: str) -> float:
        if not self.config.custom_patterns:
            return 0.0
            
        sensitivity_score = 0.0
        for pattern, weight in self.config.custom_patterns.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            sensitivity_score += matches * weight
        return sensitivity_score
    
    def save_model(self):
        try:
            model_path, vectorizer_path = self._get_cache_paths()
            joblib.dump(self.model, model_path)
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            self.logger.info(f"Model and vectorizer saved to {self.config.cache_dir}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def load_model(self) -> bool:
        try:
            model_path, vectorizer_path = self._get_cache_paths()
            if model_path.exists() and vectorizer_path.exists():
                self.model = joblib.load(model_path)
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                self.logger.info("Model and vectorizer loaded from cache")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def train(self, texts: List[str], labels: List[int], force_retrain: bool = False):
        try:
            if not force_retrain and self.load_model():
                return
            
            self._initialize_vectorizer()
            self._initialize_model()
            X = self.vectorizer.fit_transform(texts)
            self.model.fit(X, labels)
            self.save_model()
            self.logger.info("Model trained successfully")
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def predict(self, texts: Union[str, List[str]]) -> List[Dict[str, Union[int, float]]]:
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            if not self.model or not self.vectorizer:
                if not self.load_model():
                    raise ValueError("No trained model available")
            
            X = self.vectorizer.transform(texts)
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                pattern_score = self._apply_custom_patterns(texts[i])
                confidence = prob[1] + pattern_score
                
                results.append({
                    'is_sensitive': int(pred),
                    'confidence': float(confidence),
                    'model_confidence': float(prob[1]),
                    'pattern_score': float(pattern_score)
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        if not self.model or not self.vectorizer:
            if not self.load_model():
                raise ValueError("No trained model available")
                
        feature_names = self.vectorizer.get_feature_names_out()
        importances = self.model.feature_importances_
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        return feature_importance.nlargest(top_n, 'importance')