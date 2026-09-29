from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim" / "base.parquet"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

KAGGLE_DATASET = "sgpjesus/bank-account-fraud-dataset-neurips-2022"
RAW_FILE = "Base.csv"

TARGET, MONTH, AGE = "fraud_bool", "month", "customer_age"
CATEGORICAL = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
# BAF datasheet: negative values in these columns mean "missing"
SENTINEL_NEGATIVE = ["prev_address_months_count", "current_address_months_count", "bank_months_count",
                     "session_length_in_minutes", "intended_balcon_amount", "device_distinct_emails_8w"]

TRAIN_MONTHS = (0, 1, 2, 3, 4)
VALID_MONTH = 5
FUTURE_MONTHS = (6, 7)
ROLLING_FOLDS = [((0, 1, 2), 3), ((0, 1, 2, 3), 4)]
SEED = 2026

LR_GRID = (0.01, 0.1, 1.0)
XGB_GRID = {"max_depth": (3, 5), "learning_rate": (0.05, 0.1), "min_child_weight": (1, 10)}
XGB_MAX_TREES = 2000
EARLY_STOP = 50

FPR_BUDGET = 0.05
COST_RATIOS = (20, 50, 100)          # loss per missed fraud / cost of one review (assumption, no amounts in BAF)
CENTRAL_RATIO = 50
AGE_CUTOFF = 50
N_BOOT = 1000

PSI_ALERT = 0.25
FPR_BAND = (0.03, 0.07)
