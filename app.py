from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
import os



def load_data (file_name):
    csv_path = os.path.join("Data", file_name)
    return pd.read_csv(csv_path)

train_set = load_data("churn-bigml-80.csv")

# Wrapper class that drops the unneaded columns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class DropColumnTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        columns_to_drop = ['Total day minutes', 'Total eve minutes', 'Total night minutes', 'Total intl minutes']
        if all(col in X.columns for col in columns_to_drop):
            X_transformed = X.drop(['Total day minutes', 'Total eve minutes', 'Total night minutes', 'Total intl minutes'], axis=1)

        return X_transformed


# Wrapper class that handles outliers

from sklearn.base import BaseEstimator, TransformerMixin

class HandleOutliersTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        num_columns = list(X_transformed.select_dtypes(include=['number']).columns)
        to_not_remove = {'Account length', 'Customer service calls', 'Area code'}
        num_columns = [item for item in num_columns if item not in to_not_remove]
        for column in num_columns:
            Q1 = X[column].quantile(0.25)
            Q3 = X[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_fence = Q1 - 1.5 * IQR
            upper_fence = Q3 + 1.5 * IQR

            X_transformed = X_transformed[(X_transformed[column] >= lower_fence) & (X_transformed[column] <= upper_fence)]
            X_transformed = X_transformed.reset_index(drop=True)

        return X_transformed


# Wrapper class that transforms State and churn features using LabelEncoder

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np

class TransformLabelMethod(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder_dict = {}

    def fit(self, X, y=None):
        self.encoder_dict = {}
        categories = ['State', 'Churn']
        for cat in categories:
            encoder = LabelEncoder()
            encoder.fit(X[cat])
            self.encoder_dict[cat] = encoder

        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column, encoder in self.encoder_dict.items():
            new_categories = set(X[column].unique()) - set(encoder.classes_)
            if new_categories:
                all_categories = np.concatenate([encoder.classes_, list(new_categories)])
                encoder.fit(all_categories)
            X_transformed[column] = encoder.transform(X[column])

        return X_transformed


# Wrapper class for the encoder that ransforms all the remaining categorical features using OneHot encoder

from sklearn.preprocessing import OneHotEncoder

class TransformOneHotMethod(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder_dict = {}

    def fit(self, X, y=None):
        self.encoder_dict = {}
        for column in X.columns:
            if not pd.api.types.is_numeric_dtype(X[column]):
                encoder = OneHotEncoder(sparse_output=False)
                encoder.fit(X[column].values.reshape(-1, 1))
                self.encoder_dict[column] = encoder

        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column, encoder in self.encoder_dict.items():
            new_categories = set(X[column].unique()) - set(encoder.categories_[0])
            if new_categories:
                updated_categories = np.concatenate([encoder.categories_[0], list(new_categories)])
                encoder.fit(updated_categories.reshape(-1, 1))

            transformation = encoder.transform(X[column].values.reshape(-1, 1))
            modified_cat = [f"{column}({cat})" for cat in encoder.categories_[0]]
            column_transformed = pd.DataFrame(transformation, columns=modified_cat)
            X_transformed = X_transformed.drop(column, axis=1).join(column_transformed)

        return X_transformed

    
# Wrppaer class for MinMax Feature Scaling

from sklearn.preprocessing import MinMaxScaler

class MinMaxScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder_dict = {}

    def fit(self, X, y=None):
        self.encoder_dict = {}
        for column in X:
            encoder = MinMaxScaler()
            encoder.fit(X[column].values.reshape(-1, 1))
            self.encoder_dict[column] = encoder

        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column, encoder in self.encoder_dict.items():
            X_transformed[column] = encoder.transform(X_transformed[column].values.reshape(-1, 1))

        return X_transformed



# Wrapper class for feature engineering

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class GeneratingFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()

        X_transformed['Total calls'] = X['Total day calls'] + X['Total eve calls'] + X['Total night calls']
        X_transformed['Total charge'] = X['Total day charge'] + X['Total eve charge'] + X['Total night charge']
        X_transformed['Perc of customer service calls'] = (X['Customer service calls'] / X_transformed['Total calls']) * 100
        X_transformed['Day to night usage ratio'] = X['Total day calls'] / X['Total night calls']
        X_transformed['Customer service call rate perc'] = X['Customer service calls'] / X['Account length']
        X_transformed['Profitability'] = X_transformed['Total charge'] / X['Account length']

        return X_transformed



feature_engineering_pipeline = Pipeline([
    ('drop_transformer', DropColumnTransformer()),
    ('label_transformer', TransformLabelMethod()),
    ('One-Hot_transformer', TransformOneHotMethod()),
    ('feature_engineering_transformer', GeneratingFeatures()),
])
scaling_pipeline = Pipeline([
    ('scaling', MinMaxScalerTransformer())
])

train_set_transformed_f = feature_engineering_pipeline.fit_transform(train_set)
train_set_transformed_f = scaling_pipeline.fit_transform(train_set_transformed_f)


# Chemin vers le fichier du modèle
MODEL_PATH = "model.pkl"

# Charger le modèle
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
        if not hasattr(model, "predict"):
            raise ValueError("Le fichier chargé n'est pas un modèle valide.")
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    model = None

# Initialiser l'application Flask
app = Flask(__name__)

# Route pour la page d'accueil
@app.route("/")
def home():
    return render_template("home.html")  # Le fichier HTML doit être dans le dossier "templates"

@app.route("/index")
def index():
    return render_template("index.html")  # This renders the 'index.html' template

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    if model is None:
        return jsonify({"error": "Le modèle n'est pas disponible. Vérifiez le fichier model.pkl."})

    try:
        # Vérifier si un fichier a été envoyé
        if "file" not in request.files:
            return jsonify({"error": "Aucun fichier n'a été envoyé."})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Aucun fichier sélectionné."})

        # Charger le fichier CSV dans un DataFrame
        data = pd.read_csv(file)
        
        # Afficher le DataFrame dans la console
        print("Données chargées depuis le fichier CSV :")
        print(data.head())  # Affiche les 5 premières lignes du DataFrame

        if data.empty:
            return jsonify({"error": "Le fichier CSV est vide."})


        new_data = feature_engineering_pipeline.transform(data)
        new_data = scaling_pipeline.transform(new_data)

        if "Churn" in new_data.columns:
            new_data_w = new_data.drop("Churn", axis=1)


        # Effectuer les prédictions
        predictions = model.predict(new_data_w)
        if hasattr(model, "predict_proba"):
            predictions_prob = model.predict_proba(new_data_w).tolist()
        else:
            predictions_prob = None


        # Ajouter les résultats au DataFrame
        data["Prediction"] = predictions
        if predictions_prob:
            data["Probability"] = predictions_prob


        # Format the predictions as a list of dictionaries
        result = data.copy()
        result['predictions'] = predictions.tolist()
        
        # Return the result as JSON
        return jsonify(result.to_dict(orient='records'))  # Sending the prediction results as JSON

    except Exception as e:
        return jsonify({"error": str(e)})


# Lancer l'application Flask
if __name__ == "__main__":
    app.run(debug=True)
