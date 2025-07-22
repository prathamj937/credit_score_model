# Aave V2 Credit Scoring Engine

This project develops a machine learning model to assign a credit score (0-1000) to Aave V2 wallets based on their historical transaction data. The goal is to identify reliable, responsible users versus those exhibiting risky, bot-like, or exploitative behavior.

## Chosen Methodology: A Two-Stage Hybrid Model

A purely supervised learning approach is not feasible for this problem because there is no pre-existing, labeled dataset of "good" and "bad" wallets. I adopt a two-stage hybrid approach that first creates labels and then trains a model to learn them.

A key finding from the provided data sample is the **complete absence of liquidation events**. This means the most obvious indicator of risk is not available. Our strategy is therefore adapted to define risk based on more subtle, proxy indicators.

### Stage 1: Heuristic Scoring (Label Generation)

I build an interpretable, rule-based model to generate initial scores. This model defines "creditworthiness" based on proxies for financial health and responsibility.

-   **Feature Engineering:** Raw transactions are aggregated per wallet to create features.
-   **Defining Risk without Liquidations:** In place of liquidation data, I focus on:
    -   **`repay_to_borrow_ratio`**: The most heavily weighted factor. A ratio below 1 indicates a user is not fully repaying their debts, a clear sign of risk.
    -   **`deposit_to_borrow_ratio`**: A proxy for Loan-to-Value (LTV). A low ratio signifies a user is borrowing heavily against their collateral, leaving little room for price fluctuations and increasing risk.
    -   **`net_balance_usd`**: Identifies whether a user is a net supplier of capital (low risk) or a net borrower (higher risk).
-   **Weighted Scoring:** These key features are assigned weights, transformed using a `PowerTransformer` to handle skewed distributions, and combined into a final heuristic score scaled from 0-1000.
### Stage 2: Supervised Classification (Refined Prediction)

The heuristic score, while useful, is based on a simple linear combination of features. A more advanced model can capture non-linear interactions and potentially create a more robust classification system.

-   **Label Generation:** The continuous heuristic score (0-1000) is binned into discrete categories: `Risky` (0-399), `Standard` (400-799), and `Prime` (800-1000). These become the target labels for our supervised model.
-   **Model Training:** A `RandomForestClassifier` is trained on the engineered features from Stage 1 to predict these credit classes. The Random Forest is chosen for its high performance, its ability to handle feature interactions, and its built-in mechanism for calculating feature importance.
-   **Class Imbalance:** I used `class_weight='balanced'` during training to counteract the natural imbalance where "Risky" wallets are more common than "Prime" wallets.

## Processing Flow & Architecture

The `score_wallets.py` script executes the entire pipeline in one step:

1.  **Load Data:** Ingests the raw `user-wallet-transactions.json` file.
2.  **Process & Flatten:** Converts the nested JSON into a clean, tabular DataFrame of individual transactions.
3.  **Feature Engineering:** Aggregates transactions for each unique wallet, generating the feature set.
4.  **Heuristic Scoring:** Calculates the initial 0-1000 score for every wallet using the weighted formula.
5.  **Label Generation:** Bins the heuristic scores into `Risky`, `Standard`, and `Prime` classes.
6.  **Train & Evaluate:** Splits the data, trains the `RandomForestClassifier` to predict the classes, and prints a full evaluation report, including accuracy, a classification report, and feature importances.
7.  **Generate Output:** Saves a final `wallet_credit_scores.csv` file containing each wallet, its heuristic score, and its final predicted class.
8.  **Provide Analysis:** Prints a Markdown table of the score distribution, which can be used for further analysis.

## Accuracy
Model Evaluation on Test Set :
Model Accuracy: 0.9829

## Top 10 Most Important Features According to the RandomForest Model:
                feature    importance
        total_repay_usd    0.173810
  repay_to_borrow_ratio    0.130693
                n_repay    0.110265
       total_borrow_usd    0.108489
        net_balance_usd    0.102767
      total_deposit_usd    0.080338
deposit_to_borrow_ratio    0.073875
               n_borrow    0.056885
             total_txns    0.056050
      avg_txn_value_usd    0.055219
## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd aave-credit-scoring
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Place the data file:**
    Ensure your sample transaction data is named `user-wallet-transactions.json` and is located in the root directory of the project.

5.  **Run the script:**
    ```bash
    python score_wallets.py
    ```
