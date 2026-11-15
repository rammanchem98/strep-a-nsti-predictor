"""Central configuration. All data in this project is SYNTHETIC."""
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SYNTH_CSV = DATA_DIR / "nsti_gas_SYNTHETIC.csv"
GROUND_TRUTH = DATA_DIR / "ground_truth.json"
SPLIT_DIR = DATA_DIR / "splits"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ROOT / "model"
SERVED_MODEL = MODEL_DIR / "served_pipeline.joblib"
SERVED_META = MODEL_DIR / "served_meta.json"
MLRUNS_DIR = ROOT / "mlruns"
EXPERIMENT = "nsti-gas-SYNTHETIC"

N_PATIENTS = 409
TARGET = "GAS"
NUMERIC = ["AGE", "creatinine_preop", "hgb_preop"]
BINARY = ["DIABETES", "UPPER_ARM_ARRIVAL", "LOWER_ARM_ARRIVAL",
          "ANOGENITAL_ARRIVAL", "SURGERY_BEFORE"]
FEATURES = ["AGE", "DIABETES", "UPPER_ARM_ARRIVAL", "LOWER_ARM_ARRIVAL",
            "ANOGENITAL_ARRIVAL", "SURGERY_BEFORE", "creatinine_preop", "hgb_preop"]
DECOYS = ["NOISE_1", "NOISE_2", "NOISE_3"]

THRESHOLD = 0.3
DISCLAIMER = ("SYNTHETIC DATA / learning project. Not clinically validated. "
              "Must not be used for clinical decisions.")

# Paper (Katz et al., BMC Med 2026) RF pre-surgery results, quoted for comparison.
PAPER = {
    "internal_roc_auc": 0.828, "internal_roc_auc_ci": [0.763, 0.883],
    "internal_balanced_acc": 0.726, "internal_brier": 0.152, "internal_pr_auc": 0.684,
    "external_roc_auc": 0.727, "external_roc_auc_ci": [0.672, 0.783],
    "external_balanced_acc": 0.670, "external_finetuned_roc_auc": 0.758,
    "external_sens_at_0.3": 0.62, "external_spec_at_0.3": 0.67,
}
