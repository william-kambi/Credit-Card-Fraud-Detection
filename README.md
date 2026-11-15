# Credit-Card-Fraud-Detection

A binary classifier that scores card transactions for fraud risk, served as a Flask web service, packaged with Docker and deployed on AWS.


## 1. Problem description

Card fraud costs issuers, banks and merchants money on every transaction that goes undetected: the transaction amount is lost, a chargeback follows, and the customer loses trust. At the same time, fraud is rare. In this dataset only **1.7%** of transactions are fraudulent. Analysts cannot manually review every transaction, and blocking too many legitimate purchases frustrates customers and loses sales.

This project builds a machine learning model that looks at the details of a single transaction (amount, merchant category, authentication method, device, location and behaviour signals) and estimates the **probability that it is fraudulent**.

## 2. How the model will be used

The model is designed to sit inside a transaction-monitoring workflow:

1. A transaction arrives with its attributes.
2. The model, exposed as a web service, returns a fraud probability between 0 and 1.
3. The business applies a decision threshold to that score:
   - **High score**: block the transaction or hold it for urgent review.
   - **Medium score**: place it in a fraud analyst's review queue, ranked by risk.
   - **Low score**: approve automatically.

The users are fraud analysts and the payment systems that call the service. The model does not replace human judgement; it decides where analysts spend their limited time.

### Why the threshold matters

The two kinds of error have different costs:

- **Missed fraud (false negative):** the institution loses the transaction amount plus chargeback and investigation costs.
- **False alarm (false positive):** a legitimate customer is declined, or an analyst spends time on a clean transaction.

Because fraud is rare, **accuracy is misleading**: a model that labels everything "legitimate" would be 98.3% accurate and catch no fraud. This project therefore evaluates models with **PR-AUC, precision and recall** (plus ROC-AUC for reference), and chooses the decision threshold from the trade-off between the two error costs rather than using the default 0.5 cutoff.

## 3. Dataset

- **Source:** Kaggle, [`uditjain13/credit-card-fraud-detection-2026`](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026)
- **File:** `data/credit_card_fraud_2026.csv` (committed to this repository)
- **Size:** 20,000 transactions, 26 columns
- **Target:** `is_fraud` (1 = fraud, 0 = legitimate); 339 frauds (1.7%)
- **Data quality:** no missing values, no duplicate rows

### Re-downloading the data

The CSV is already in `data/`. To fetch it again from Kaggle:

**Option 1: manual.** Open the dataset page above, click **Download**, unzip, and place the CSV in `data/`.

**Option 2: Kaggle CLI.**

```bash
pip install kaggle
# save your API token (Kaggle > Settings > Create New Token) to ~/.kaggle/kaggle.json
kaggle datasets download -d uditjain13/credit-card-fraud-detection-2026 -p data --unzip
```

### Features

| Group | Columns |
|---|---|
| Categorical | `merchant_category`, `card_type`, `auth_method`, `channel`, `device_type` |
| Boolean flags | `is_foreign_transaction`, `is_new_merchant`, `used_vpn`, `ip_country_mismatch`, `billing_shipping_mismatch`, `is_ai_generated_scam_attempt` |
| Numeric | `amount_usd`, `hours_since_last_txn`, `txn_count_last_24h`, `distance_from_home_km`, `card_age_months`, `customer_age`, `account_balance_usd`, `cvv_retry_count`, `velocity_score`, `time_of_day_hour`, `day_of_week`, `merchant_risk_score`, `prior_disputes` |
| Dropped | `transaction_id` (identifier, no predictive value) |

## 4. Exploratory data analysis: key findings

The full analysis is in `notebook.ipynb`. Highlights:

- **No target leakage.** The strongest single feature (`merchant_risk_score`) reaches only about 0.71 ROC-AUC on its own, so no column gives the answer away.
- **Strongest fraud signals** (overall fraud rate is 1.7%):

  | Signal | Fraud rate |
  |---|---|
  | `is_ai_generated_scam_attempt` = True | 9.1% (vs 1.6%) |
  | `cvv_retry_count` = 2 | 16% (0 retries: 0.9%) |
  | `ip_country_mismatch` = True | 7.8% (vs 1.3%) |
  | `billing_shipping_mismatch` = True | 7.8% (vs 1.4%) |
  | `is_foreign_transaction` = True | 5.9% (vs 1.4%) |
  | `auth_method` = No Authentication | 4.4% (Biometric: 0.8%) |
  | `merchant_category` = Crypto Exchange / Gift Cards | 5.0% / 4.4% |

- **Weak features:** `card_type`, `channel`, `device_type`, `day_of_week`, `distance_from_home_km`, `account_balance_usd` and `customer_age` show little separation.
- **Class imbalance** (98.3% / 1.7%) shapes every later choice: stratified splits, PR-AUC as the main metric, and threshold tuning.

## 5. Modelling approach

1. **Data preparation:** drop `transaction_id`, convert boolean flags to integers, one-hot encode categoricals with `DictVectorizer`.
2. **Split:** stratified train / validation / test (60 / 20 / 20) with a fixed random seed. Stratified 5-fold cross-validation is used for model comparison, because only 339 frauds are available.
3. **Models compared:** logistic regression (baseline), random forest, XGBoost.
4. **Tuning:** small hyperparameter grids, scored with PR-AUC on cross-validation folds.
5. **Selection:** the final model is chosen on PR-AUC and on recall at a fixed precision, then evaluated once on the held-out test set.
6. **Threshold:** chosen from the cost trade-off between missed fraud and false alarms.

### Results

> Fill in after training.

| Model | CV PR-AUC | Validation PR-AUC | Validation ROC-AUC |
|---|---|---|---|
| Logistic regression | | | |
| Random forest | | | |
| XGBoost | | | |

Final model: **[name]**, test PR-AUC **[value]**, chosen threshold **[value]** (precision **[value]**, recall **[value]**).

## 6. Repository structure

```
.
├── README.md
├── data/
│   └── credit_card_fraud_2026.csv
├── notebook.ipynb        # data prep, EDA, feature importance, model selection, tuning
├── train.py              # trains the final model and saves model.bin
├── predict.py            # Flask web service that loads model.bin
├── model.bin             # trained model (DictVectorizer + classifier + threshold)
├── requirements.txt      # pinned dependencies
├── Dockerfile            # container for the web service
└── deployment/           # screenshots / demo video of the deployed service
```

## 7. How to run the project

### Prerequisites

- Python 3.11 or newer (match the version in the `Dockerfile`)
- Docker (for the containerised service)
- Git

### 7.1 Clone and set up the environment

```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 7.2 Explore the notebook (optional)

```bash
jupyter notebook notebook.ipynb
```

### 7.3 Train the final model

```bash
python train.py
```

This reads `data/credit_card_fraud_2026.csv`, trains the selected model with the tuned parameters, prints validation and test metrics, and writes `model.bin`.

### 7.4 Run the web service locally (without Docker)

```bash
python predict.py                              # development server
# or, closer to production:
gunicorn --bind 0.0.0.0:9696 predict:app
```

### 7.5 Test the service

With the service running, send a sample transaction:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount_usd": 640.0,
    "merchant_category": "Gift Cards",
    "card_type": "Visa",
    "auth_method": "No Authentication",
    "channel": "Online",
    "device_type": "Android Phone",
    "is_foreign_transaction": 1,
    "hours_since_last_txn": 0.4,
    "txn_count_last_24h": 7,
    "distance_from_home_km": 120.5,
    "card_age_months": 30,
    "customer_age": 41,
    "account_balance_usd": 1500.0,
    "is_new_merchant": 1,
    "used_vpn": 1,
    "ip_country_mismatch": 1,
    "billing_shipping_mismatch": 1,
    "cvv_retry_count": 2,
    "velocity_score": 55.0,
    "time_of_day_hour": 3,
    "day_of_week": 5,
    "is_ai_generated_scam_attempt": 1,
    "merchant_risk_score": 78.0,
    "prior_disputes": 1
  }'
```

Expected response shape:

```json
{"fraud_probability": 0.0, "is_fraud": false}
```

The values above are placeholders; the real numbers depend on the trained model.

### 7.6 Run with Docker

```bash
docker build -t fraud-detection .
docker run -it --rm -p 9696:9696 fraud-detection
```

Then repeat the `curl` command from 7.5 against `http://localhost:9696/predict`.

## 8. API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/ping` | Health check. Returns `PONG`. |
| `POST` | `/predict` | Takes one transaction as JSON, returns the fraud probability and the decision at the configured threshold. |

**Request body:** a JSON object with the 24 feature fields listed in [Features](#features) (all except `transaction_id`). Boolean flags are sent as `0` or `1`.

**Response:**

| Field | Type | Meaning |
|---|---|---|
| `fraud_probability` | float, 0 to 1 | Model-estimated probability of fraud |
| `is_fraud` | boolean | `true` if the probability is at or above the chosen threshold |

## 9. Deployment to AWS

The Docker image is deployed on AWS.

- **Service used:** [AWS Elastic Beanstalk / ECS Fargate: fill in]
- **Public URL:** [fill in, or see the screenshots below if the service has been taken down to avoid charges]
- **Evidence of the deployed service:** see `deployment/` for [screenshots / a short screen recording] showing a request sent to the live endpoint and the response.

### Deployment steps

> Fill in with the exact commands used, for example:
> 1. Build and tag the image.
> 2. Push it to Amazon ECR (or deploy the Dockerfile directly with Elastic Beanstalk).
> 3. Create the service and set the container port to 9696.
> 4. Test the public endpoint with `curl`.

Never commit AWS credentials to this repository.

## 10. Limitations

- **Few positive cases.** With 339 frauds, evaluation metrics have wide uncertainty. Cross-validation and small tuning grids reduce, but do not remove, this.
- **Likely synthetic data.** The patterns appear generated, so reported metrics demonstrate the pipeline rather than real-world performance.
- **Feature availability.** Fields such as `is_ai_generated_scam_attempt` and `merchant_risk_score` must be available at the time a transaction is scored. This project assumes an upstream system provides them.
- **No drift or temporal validation.** The dataset has no timestamp suitable for a time-based split, so performance over time was not tested.
- **Threshold is cost-assumption based.** The chosen threshold depends on assumed costs of missed fraud versus false alarms; a real deployment would calibrate it with actual business costs.


