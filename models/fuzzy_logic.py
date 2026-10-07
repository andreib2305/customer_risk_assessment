def calculate_fuzzy_risk(row):
    risk = 0.0
    if row['payment_delay_days'] > 15:
        risk += 40
    elif row['payment_delay_days'] > 5:
        risk += 20

    if row['complaints'] >= 2:
        risk += 30
    elif row['complaints'] == 1:
        risk += 15

    if row['service_score'] <= 4:
        risk += 30
    elif row['service_score'] <= 6:
        risk += 15

    if row['tariff'] == 'premium':
        risk *= 0.85

    return min(float(risk), 100.0)