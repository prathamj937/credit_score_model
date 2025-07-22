import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

filepath = "user-wallet-transactions.json"

def load_json(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"FATAL: The data file '{filepath}' was not found.")
        exit()

def process_transactions(raw_data):
    processed = []
    for tx in tqdm(raw_data, desc="Processing Transactions"):
        action_data = tx.get('actionData', {}) or {}
        try:
            raw_amt = float(action_data.get('amount', 0)) / 1e18
        except (ValueError, TypeError):
            raw_amt = 0.0
        try:
            usd_price = float(action_data.get('assetPriceUSD', 0))
        except (ValueError, TypeError):
            usd_price = 0.0
        processed.append({
            'wallet': tx.get('userWallet'),
            'action': tx.get('action', '').lower(),
            'timestamp': tx.get('timestamp'),
            'asset': action_data.get('assetSymbol', 'UNKNOWN'),
            'usd_value': usd_price * raw_amt,
        })
    return pd.DataFrame(processed)

def engineer_features(df):
    if df.empty:
        return pd.DataFrame()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.sort_values(by='timestamp', inplace=True)

    def agg_wallet(group):
        deposits = group[group['action'] == 'deposit']['usd_value']
        borrows = group[group['action'] == 'borrow']['usd_value']
        repays = group[group['action'] == 'repay']['usd_value']
        liquidations = group[group['action'] == 'liquidationcall']['usd_value']
        total_deposit, total_borrow, total_repay, total_liquidation = (
            deposits.sum(), borrows.sum(), repays.sum(), liquidations.sum()
        )
        return pd.Series({
            'total_txns': len(group),
            'unique_assets': group['asset'].nunique(),
            'active_days': group['timestamp'].dt.date.nunique(),
            'total_deposit_usd': total_deposit,
            'total_borrow_usd': total_borrow,
            'total_repay_usd': total_repay,
            'total_liquidated_usd': total_liquidation,
            'net_balance_usd': total_deposit - total_borrow,
            'n_deposit': deposits.count(),
            'n_borrow': borrows.count(),
            'n_repay': repays.count(),
            'n_liquidation': liquidations.count(),
            'has_been_liquidated': int(total_liquidation > 0),
            'avg_txn_value_usd': group['usd_value'].mean(),
            'repay_to_borrow_ratio': total_repay / total_borrow if total_borrow > 0 else 0,
            'deposit_to_borrow_ratio': total_deposit / total_borrow if total_borrow > 0 else 0,
            'liquidation_severity_ratio': total_liquidation / total_borrow if total_borrow > 0 else 0,
            'first_txn': group['timestamp'].min(),
            'last_txn': group['timestamp'].max(),
        })

    features = df.groupby('wallet').apply(agg_wallet).reset_index()
    features['wallet_age_days'] = (features['last_txn'] - features['first_txn']).dt.days + 1
    features.drop(['first_txn', 'last_txn'], axis=1, inplace=True)
    features.fillna(0, inplace=True)
    features.replace([np.inf, -np.inf], 0, inplace=True)
    return features

def calculate_heuristic_score(df_features):
    features = df_features.drop('wallet', axis=1)

    weights = {
        'repay_to_borrow_ratio': 0.4,
        'deposit_to_borrow_ratio': 0.3,
        'net_balance_usd': 0.15,
        'wallet_age_days': 0.1,
        'active_days': 0.1,
        'total_deposit_usd': 0.05,
        'total_txns': 0.05,
        'unique_assets': 0.05,
    }

    score_features = [col for col in features.columns if col in weights]
    X = features[score_features].copy()

    pt = PowerTransformer(method='yeo-johnson')
    X_transformed = pt.fit_transform(X)
    X_transformed = pd.DataFrame(X_transformed, columns=score_features)

    raw_score = sum(X_transformed[col] * weight for col, weight in weights.items())

    scaler = MinMaxScaler(feature_range=(0, 1000))
    final_scores = scaler.fit_transform(raw_score.values.reshape(-1, 1)).flatten()

    scored_df = df_features.copy()
    scored_df['heuristic_score'] = final_scores.round().astype(int)
    return scored_df

def create_labels_and_train_model(scored_df):
    bins = [-1, 399, 799, 1001]
    labels = ["Risky", "Standard", "Prime"]
    scored_df['credit_class'] = pd.cut(scored_df['heuristic_score'], bins=bins, labels=labels)

    print("\nSTAGE 2: SUPERVISED MODEL")
    print("\nI created the following distribution of credit classes:")
    print(scored_df['credit_class'].value_counts())

    X = scored_df.drop(['wallet', 'heuristic_score', 'credit_class'], axis=1)
    y = scored_df['credit_class']
    X.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"\nI split the data into {len(X_train)} training samples and {len(X_test)} testing samples.")
    print("\nI am now training the RandomForestClassifier...")

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    print("Model training completed successfully.")
    print("\nModel Evaluation on Test Set")

    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features According to the RandomForest Model:")
    print(feature_importances.head(10).to_string(index=False))

    scored_df['predicted_class'] = model.predict(X)
    return scored_df, model

def generate_analysis_data(df):
    print("\n\nFOR ANALYSIS.MD")
    score_bins = range(0, 1001, 100)
    labels = [f"{i}-{i+99}" for i in range(0, 900, 100)] + ["900-1000"]
    df['score_range'] = pd.cut(df['heuristic_score'], bins=score_bins, labels=labels, right=False)
    distribution = df['score_range'].value_counts().sort_index().reset_index()
    distribution.columns = ['Score Range', 'Number of Wallets']
    print("\nGenerated the Heuristic Score Distribution:")
    print(distribution.to_markdown(index=False))

def main(json_path):
    raw_data = load_json(json_path)
    print("STAGE 1: HEURISTIC SCORING")
    df_tx = process_transactions(raw_data)
    df_features = engineer_features(df_tx)
    print(f"I engineered features for {df_features.shape[0]} unique wallets.")

    if df_features.empty:
        print("No features were generated. Exiting.")
        return

    scored_df = calculate_heuristic_score(df_features)
    print("I generated revised heuristic scores for all wallets.")
    final_df, model = create_labels_and_train_model(scored_df)

    output_cols = ['wallet', 'heuristic_score', 'predicted_class']
    final_df[output_cols].to_csv("wallet_credit_scores.csv", index=False)
    print("\nI saved the final scores and predictions to wallet_credit_scores.csv")
    generate_analysis_data(final_df)

if __name__ == "__main__":
    main(filepath)