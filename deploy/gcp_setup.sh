#!/usr/bin/env bash
# One-time GCP setup for Cloud Run + Artifact Registry + Workload Identity Federation.
# NOT run automatically. Review, fill in the variables, then run it yourself after `gcloud auth login`.
# Creates billable resources (Artifact Registry storage, Cloud Run when invoked). No JSON keys are created.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"                # e.g. nsti-gas-mlops-<suffix>
BILLING_ACCOUNT="${BILLING_ACCOUNT:?set BILLING_ACCOUNT}" # gcloud billing accounts list
REGION="${REGION:-europe-west4}"
GITHUB_REPO="${GITHUB_REPO:-rammanchem98/strep-a-nsti-predictor}"
BUDGET_AMOUNT="${BUDGET_AMOUNT:-10USD}"  # MUST use the billing account's currency (e.g. 8GBP), else INVALID_ARGUMENT

DEPLOY_SA="gh-deployer@${PROJECT_ID}.iam.gserviceaccount.com"
RUNTIME_SA="nsti-runtime@${PROJECT_ID}.iam.gserviceaccount.com"
POOL="github"
PROVIDER="github-oidc"

# 1. Project + billing (skip create if the project already exists)
gcloud projects describe "$PROJECT_ID" >/dev/null 2>&1 || gcloud projects create "$PROJECT_ID"
gcloud billing projects link "$PROJECT_ID" --billing-account "$BILLING_ACCOUNT"
gcloud config set project "$PROJECT_ID"

# 2. APIs
gcloud services enable run.googleapis.com artifactregistry.googleapis.com iamcredentials.googleapis.com \
  sts.googleapis.com cloudbilling.googleapis.com billingbudgets.googleapis.com

# 3. Billing alert (email to billing admins at 50/90/100%)
gcloud billing budgets create --billing-account "$BILLING_ACCOUNT" \
  --display-name "nsti-gas-mlops budget" --budget-amount "$BUDGET_AMOUNT" \
  --filter-projects "projects/${PROJECT_ID}" \
  --threshold-rule percent=0.5 --threshold-rule percent=0.9 --threshold-rule percent=1.0

# 4. Artifact Registry repo
gcloud artifacts repositories create nsti --repository-format docker --location "$REGION" \
  --description "NSTI GAS API images (SYNTHETIC data)"

# 5. Service accounts: deployer (used by GitHub) and minimal runtime identity
gcloud iam service-accounts create gh-deployer --display-name "GitHub Actions deployer"
gcloud iam service-accounts create nsti-runtime --display-name "Cloud Run runtime (no roles)"
for role in roles/run.admin roles/artifactregistry.writer; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:${DEPLOY_SA}" --role "$role" --condition=None
done
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" \
  --member "serviceAccount:${DEPLOY_SA}" --role roles/iam.serviceAccountUser

# 6. Workload Identity Federation, restricted to this one GitHub repo
gcloud iam workload-identity-pools create "$POOL" --location global --display-name "GitHub"
gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" --location global \
  --workload-identity-pool "$POOL" --issuer-uri "https://token.actions.githubusercontent.com" \
  --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
  --attribute-condition "assertion.repository=='${GITHUB_REPO}'"
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format 'value(projectNumber)')
gcloud iam service-accounts add-iam-policy-binding "$DEPLOY_SA" --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/attribute.repository/${GITHUB_REPO}"

cat <<EOF

Done. Set these GitHub repository VARIABLES (Settings -> Secrets and variables -> Actions -> Variables):
  GCP_PROJECT_ID   = ${PROJECT_ID}
  GCP_REGION       = ${REGION}
  GCP_WIF_PROVIDER = projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}
  GCP_DEPLOY_SA    = ${DEPLOY_SA}
  GCP_RUNTIME_SA   = ${RUNTIME_SA}
(None of these are secrets; no keys are stored in GitHub.)
EOF
