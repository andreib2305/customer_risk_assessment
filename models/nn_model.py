from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

numerical_features = ['age', 'income', 'months_with_company', 'complaints', 'service_score', 'payment_delay_days']
categorical_features = ['tariff', 'channel']
hybrid_numerical_features = numerical_features + ['fuzzy_risk_index']

preprocessor_base = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

preprocessor_hybrid = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), hybrid_numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

def get_base_pipeline():
    return Pipeline(steps=[
        ('preprocessor', preprocessor_base),
        ('classifier', MLPClassifier(hidden_layer_sizes=(32, 16), solver='lbfgs', max_iter=300, random_state=42))
    ])

def get_hybrid_pipeline():
    return Pipeline(steps=[
        ('preprocessor', preprocessor_hybrid),
        ('classifier', MLPClassifier(hidden_layer_sizes=(32, 16), solver='lbfgs', max_iter=300, random_state=42))
    ])