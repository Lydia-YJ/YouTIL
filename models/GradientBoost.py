import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as SKGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
import joblib

class GradientBoostClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.best_params = None
        
    def build_model(self):
        """Gradient Boosting 모델 구축"""
        self.model = SKGradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            max_features='sqrt'
        )
        return self.model
    
    def train(self, X_train, y_train, use_grid_search=False):
        """모델 훈련"""
        if self.model is None:
            self.build_model()
        
        # 클래스 불균형을 위한 샘플 가중치 계산
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=y_train
        )
        
        # 각 샘플의 가중치 계산
        sample_weights = np.array([class_weights[y] for y in y_train])
        
        if use_grid_search:
            # 그리드 서치로 하이퍼파라미터 최적화
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            grid_search = GridSearchCV(
                self.model,
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train, sample_weight=sample_weights)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"최적 파라미터: {self.best_params}")
            print(f"최적 F1 스코어: {grid_search.best_score_:.4f}")
        else:
            # 기본 훈련
            self.model.fit(X_train, y_train, sample_weight=sample_weights)
        
        return self.model
    
    def predict(self, X_test):
        """예측"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """확률 예측"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        return self.model.predict_proba(X_test)[:, 1]
    
    def get_feature_importance(self):
        """특성 중요도 반환"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        return self.model.feature_importances_
    
    def save_model(self, filepath):
        """모델 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """모델 로드"""
        self.model = joblib.load(filepath)
    
    def get_model_info(self):
        """모델 정보 출력"""
        if self.model is None:
            return "모델이 구축되지 않았습니다."
        
        info = {
            'n_estimators': self.model.n_estimators,
            'learning_rate': self.model.learning_rate,
            'max_depth': self.model.max_depth,
            'min_samples_split': self.model.min_samples_split,
            'min_samples_leaf': self.model.min_samples_leaf,
            'subsample': self.model.subsample,
            'best_params': self.best_params
        }
        return info