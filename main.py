import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data.data_generator import generate_dataset
from models.fuzzy_logic import calculate_fuzzy_risk
from models.nn_model import get_base_pipeline, get_hybrid_pipeline

def get_recommendation(probability):
    if probability > 0.6:
        return "Высокий риск оттока! Рекомендуется незамедлительно закрепить за клиентом персонального менеджера, предложить индивидуальную скидку на тариф или бесплатный апгрейд сервиса, а также оперативно решить зафиксированные проблемы/жалобы."
    elif probability > 0.3:
        return "Умеренный риск оттока. Стоит направить клиенту персонализированное предложение, информировать о новых полезных функциях продукта и провести короткий опрос удовлетворенности качеством сервиса."
    else:
        return "Низкий риск оттока. Клиент стабилен. Рекомендуется включить его в стандартные программы лояльности для поддержания высокого уровня вовлеченности."

def main():
    df = generate_dataset(n_samples=1500)

    X_base = df.drop(columns=['churn', 'fuzzy_risk_index'])
    X_hybrid = df.drop(columns=['churn'])
    y = df['churn']

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_base, y, test_size=0.2, random_state=42)
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_hybrid, y, test_size=0.2, random_state=42)

    model_base = get_base_pipeline()
    model_base.fit(X_train_b, y_train_b)

    model_hybrid = get_hybrid_pipeline()
    model_hybrid.fit(X_train_h, y_train_h)

    y_pred_b = model_base.predict(X_test_b)
    y_pred_h = model_hybrid.predict(X_test_h)

    metrics_comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
        'Without Fuzzy Index': [
            accuracy_score(y_test_b, y_pred_b),
            precision_score(y_test_b, y_pred_b, zero_division=0),
            recall_score(y_test_b, y_pred_b, zero_division=0),
            f1_score(y_test_b, y_pred_b, zero_division=0)
        ],
        'Hybrid (With Fuzzy Index)': [
            accuracy_score(y_test_h, y_pred_h),
            precision_score(y_test_h, y_pred_h, zero_division=0),
            recall_score(y_test_h, y_pred_h, zero_division=0),
            f1_score(y_test_h, y_pred_h, zero_division=0)
        ]
    })

    print(metrics_comparison.to_string(index=False))

    joblib.dump(model_hybrid, 'best_churn_model.pkl')
    loaded_model = joblib.load('best_churn_model.pkl')

    new_client = pd.DataFrame([{
        'age': 42,
        'income': 90000,
        'months_with_company': 8,
        'complaints': 2,
        'service_score': 3,
        'payment_delay_days': 12,
        'tariff': 'standard',
        'channel': 'web'
    }])

    new_client['fuzzy_risk_index'] = calculate_fuzzy_risk(new_client.iloc[0])

    churn_probability = loaded_model.predict_proba(new_client)[0][1]
    print(f"\nВероятность оттока нового клиента: {churn_probability * 100:.1f}%")

    recommendation = get_recommendation(churn_probability)
    print(f"Управленческая рекомендация: {recommendation}")
    new_client_result = new_client.copy()
    new_client_result['churn_probability'] = round(churn_probability, 4)
    new_client_result['recommendation'] = recommendation
    new_client_result.to_csv('new_client_prediction.csv', index=False, encoding='utf-8-sig')

    print("\nФайлы 'metrics_comparison.csv' и 'new_client_prediction.csv' успешно сохранены.")

if __name__ == '__main__':
    main()