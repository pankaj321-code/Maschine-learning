
# 🔹 Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 🔹 Model Training Function
def train_models(X, y, dataset_name):
    print(f"\n🧪 Dataset: {dataset_name}")

    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define Models
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(),
        'XGBoost': XGBClassifier(eval_metric='logloss')
    }

    # Train and Evaluate
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\n🔹 Model: {name}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(classification_report(y_test, y_pred))

# 🔹 Heart Disease Dataset
url_heart = "https://raw.githubusercontent.com/rahulrajpl/Heart-Disease-UCI-Dataset/master/heart.csv"
heart = pd.read_csv(url_heart)
X_heart = heart.drop('target', axis=1)
y_heart = heart['target']
train_models(X_heart, y_heart, "Heart Disease")

# 🔹 Diabetes Dataset (Pima Indians Diabetes)
url_diabetes = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
diabetes = pd.read_csv(url_diabetes)
X_diabetes = diabetes.drop('Outcome', axis=1)
y_diabetes = diabetes['Outcome']
train_models(X_diabetes, y_diabetes, "Diabetes")

# 🔹 Breast Cancer Dataset (sklearn)
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y_cancer = pd.Series(cancer.target)
train_models(X_cancer, y_cancer, "Breast Cancer")
