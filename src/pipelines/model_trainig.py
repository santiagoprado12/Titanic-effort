# Data processing
import numpy as np
import pandas as pd

# sklearn Pipelines
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV

# custom pipelines
from src.pipelines.feature_selection import FeatureSelection

#save model
from joblib import dump



class PipelineBuilding:

    def __init__(self, X:pd.DataFrame, y:pd.DataFrame, atributes_types:dict) -> None:

        self.X = X
        self.y = y
        self.atributes_types = atributes_types
        cat_columns = self._columns_after_processing(X, atributes_types["categorical_features"])
        self.all_columns = atributes_types["numeric_features"] + atributes_types["ordinal_attributes"] + cat_columns


    def _columns_after_processing(self, titanic_data: pd.DataFrame, categorical_features: list):

        categorical_features_columns = []

        for col in categorical_features:
            for x in range(0, len(titanic_data[col].dropna().unique())):
                categorical_features_columns.extend([col + str(x)])

        return categorical_features_columns
    

    def build_full_pipeline(self):

        atributes_types = self.atributes_types

        processing_feature_selection_pipeline = self._build_data_processing_pipeline(
        atributes_types["numeric_features"], atributes_types["ordinal_attributes"], 
        atributes_types["categorical_features"])

        proc_and_feat_select_pipeline = self._build_processing_and_feature_selection_pipeline(
            processing_feature_selection_pipeline, self.all_columns)
        
        models_pipelines = self._build_create_model_pipeline(
            proc_and_feat_select_pipeline)
            
        return models_pipelines
    

    def _build_data_processing_pipeline(self, numeric_features: list, ordinal_attributes: list, categorical_features: list) -> Pipeline:

        num_processing_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
        ])

        cat_processing_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="most_frequent")),
            ('cat', OneHotEncoder()),
        ])

        data_processing_pipeline = ColumnTransformer([
            ("num", num_processing_pipeline, numeric_features),
            ("ord", OrdinalEncoder(), ordinal_attributes),
            ("cat", cat_processing_pipeline,
            categorical_features),
        ])

        return data_processing_pipeline


    def _build_processing_and_feature_selection_pipeline(self, data_processing_pipeline: Pipeline, new_columns: list) -> Pipeline:

        processing_feature_selection_pipeline = Pipeline([
            ("data_processing", data_processing_pipeline),
            ("FeatureSelection", FeatureSelection(
                columns=new_columns))
        ])

        return processing_feature_selection_pipeline


    def _build_create_model_pipeline(self, processing_feature_selection_pipeline: Pipeline) -> dict:

        param_grid_rf = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False],
            'random_state': [42]
        }

        param_grid_gb = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 3],
            'subsample': [0.8, 0.9, 1.0],
            'random_state': [42]
        }

        param_grid_knn = {
            'n_neighbors': np.arange(1, 21),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }

        rf_random_search = RandomizedSearchCV(RandomForestClassifier(),
                                            param_distributions=param_grid_rf, n_iter=15, 
                                            cv=5, verbose=0, random_state=42, n_jobs=-1)
        gb_random_search = RandomizedSearchCV(GradientBoostingClassifier(),
                                            param_distributions=param_grid_gb, n_iter=15, 
                                            cv=5, verbose=0, random_state=42, n_jobs=-1)
        knn_random_search = RandomizedSearchCV(KNeighborsClassifier(),
                                            param_distributions=param_grid_knn, n_iter=15, 
                                            cv=5, verbose=0, random_state=42, n_jobs=-1)

        full_pipeline_rf = Pipeline([
            ('preparation', processing_feature_selection_pipeline),
            ('model', rf_random_search)
        ])

        full_pipeline_gb = Pipeline([
            ('preparation', processing_feature_selection_pipeline),
            ('model', gb_random_search)
        ])

        full_pipeline_knn = Pipeline([
            ('preparation', processing_feature_selection_pipeline),
            ('model', knn_random_search)
        ])

        return {"random_forest": full_pipeline_rf,
                "gradient_boosting": full_pipeline_gb, 
                "knn": full_pipeline_knn}
    

class ModelTraining(PipelineBuilding):

    def __init__(self, X:pd.DataFrame, y:pd.DataFrame, atributes_types:dict, models:list) -> None:
        
        super().__init__(X, y, atributes_types)
        models_pipelines = self.build_full_pipeline()
        self.X = X
        self.y = y
        self.models = {}
        for name in models:
            self.models[name] = models_pipelines[name] 
    
    def train_models(self) -> dict:

        for name in self.models:
            self.models[name].fit(self.X, self.y)
        
        return self.models
    
    def generate_scores(self, X_test: pd.DataFrame, y_test:pd.DataFrame) -> dict:

        scores = {}
        for model in self.models:
            scores[model] = self.models[model].score(X_test, y_test)
        
        return scores
    
    def best_model(self, X_test: pd.DataFrame, y_test:pd.DataFrame) -> str:

        scores = self.generate_scores(X_test, y_test)
        best_model = max(scores, key=scores.get)

        return best_model
    
    def save_model(self, model_name: str, path: str) -> None:
        """Save a model to a given path"""

        model = self.models[model_name]
        dump(model, path)

    


    

