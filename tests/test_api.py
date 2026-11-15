import pytest
from fastapi.testclient import TestClient

from src.api import app

VALID = {"AGE": 45, "DIABETES": 0, "UPPER_ARM_ARRIVAL": 1, "LOWER_ARM_ARRIVAL": 0,
         "ANOGENITAL_ARRIVAL": 0, "SURGERY_BEFORE": 0, "creatinine_preop": 180.0, "hgb_preop": 12.1}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["data"] == "SYNTHETIC"


def test_predict_valid(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability_gas"] <= 1
    assert body["gas_flag"] == (body["probability_gas"] >= 0.3)
    assert body["threshold"] == 0.3
    assert body["data"] == "SYNTHETIC"
    assert "not be used for clinical decisions" in body["disclaimer"]


def test_predict_missing_labs_imputed(client):
    r = client.post("/predict", json={**VALID, "creatinine_preop": None, "hgb_preop": None})
    assert r.status_code == 200


def test_directionality(client):
    """Sanity: high creatinine + upper arm should score above recent surgery + anogenital."""
    hi = client.post("/predict", json={**VALID, "creatinine_preop": 400.0}).json()["probability_gas"]
    lo = client.post("/predict", json={**VALID, "UPPER_ARM_ARRIVAL": 0, "ANOGENITAL_ARRIVAL": 1,
                                       "SURGERY_BEFORE": 1, "creatinine_preop": 60.0}).json()["probability_gas"]
    assert hi > lo


@pytest.mark.parametrize("patch", [{"AGE": 10}, {"AGE": 120}, {"DIABETES": 2}, {"creatinine_preop": -5},
                                   {"hgb_preop": 40}, {"LOWER_ARM_ARRIVAL": 1}, {"EXTRA": 1}])
def test_predict_rejects_invalid(client, patch):
    r = client.post("/predict", json={**VALID, **patch})
    assert r.status_code == 422


def test_predict_missing_field(client):
    body = dict(VALID)
    body.pop("AGE")
    assert client.post("/predict", json=body).status_code == 422
