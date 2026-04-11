"""
Reads all runs from the local mlruns/ file store and re-logs them
to the EC2 MLflow server at http://98.93.2.225:5000
"""
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

LOCAL_URI = f"file://{Path('mlruns').resolve()}"
EC2_URI   = "http://98.93.2.225:5000"
EXP_NAME  = "covered-call-strategy-classification"

# ── Read from local ───────────────────────────────────────────────────────────
local_client = MlflowClient(tracking_uri=LOCAL_URI)
local_exp    = local_client.get_experiment_by_name(EXP_NAME)

if not local_exp:
    print("No local experiment found. Run notebook 10 first.")
    exit(1)

local_runs = local_client.search_runs(
    experiment_ids=[local_exp.experiment_id],
    order_by=["attributes.start_time ASC"]
)
print(f"Found {len(local_runs)} local runs to push\n")

# ── Write to EC2 ──────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(EC2_URI)
ec2_client = MlflowClient(tracking_uri=EC2_URI)

# Get or create experiment on EC2
ec2_exp = ec2_client.get_experiment_by_name(EXP_NAME)
if ec2_exp:
    ec2_exp_id = ec2_exp.experiment_id
    print(f"Experiment already exists on EC2 (id={ec2_exp_id})")
else:
    ec2_exp_id = ec2_client.create_experiment(EXP_NAME)
    print(f"Created experiment on EC2 (id={ec2_exp_id})")

# Check which runs already exist on EC2
existing_runs = ec2_client.search_runs(experiment_ids=[ec2_exp_id])
existing_names = set(r.data.tags.get("mlflow.runName", "") for r in existing_runs)
print(f"Already on EC2: {existing_names}\n")

pushed = 0
skipped = 0

for run in local_runs:
    run_name = run.data.tags.get("mlflow.runName", run.info.run_id[:8])

    # Skip registry-only runs (no metrics logged)
    if not run.data.metrics:
        print(f"  Skipping {run_name} — no metrics")
        skipped += 1
        continue

    # Skip duplicates (same name already on EC2)
    if run_name in existing_names:
        print(f"  Skipping {run_name} — already exists on EC2")
        skipped += 1
        continue

    print(f"  Pushing: {run_name}")

    with mlflow.start_run(experiment_id=ec2_exp_id, run_name=run_name) as ec2_run:
        # Log params
        for k, v in run.data.params.items():
            mlflow.log_param(k, v)

        # Log metrics
        for k, v in run.data.metrics.items():
            mlflow.log_metric(k, v)

        # Log tags (skip internal mlflow tags)
        for k, v in run.data.tags.items():
            if not k.startswith("mlflow."):
                mlflow.set_tag(k, v)

        # Log model_type tag if present
        model_type = run.data.tags.get("model_type", "")
        if model_type:
            mlflow.set_tag("model_type", model_type)

    pushed += 1

print(f"\nDone — pushed {pushed} runs, skipped {skipped}")
print(f"View at: {EC2_URI}")
