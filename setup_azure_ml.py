from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml.entities import Environment
from azure.ai.ml import command
from azure.ai.ml import UserIdentityConfiguration
from azure.ai.ml import Input

import os

credential = DefaultAzureCredential()

# Client initialization
ml_client = MLClient(
    credential=credential,
    subscription_id="75c73b60-6890-4ba6-9a83-83a1589ae4e5",
    resource_group_name="tristan.retali-rg",
    workspace_name="health-chatbot-training",
)
print(ml_client)

gpu_compute_target = "gpu-cluster"

try:
    # let's see if the compute target already exists
    gpu_cluster = ml_client.compute.get(gpu_compute_target)
    print(
        f"You already have a cluster named {gpu_compute_target}, we'll reuse it as is."
    )

except Exception:
    print("Creating a new gpu compute target...")

    # Let's create the Azure ML compute object with the intended parameters
    gpu_cluster = AmlCompute(
        # Name assigned to the compute cluster
        name=gpu_compute_target,
        # Azure ML Compute is the on-demand VM service
        type="amlcompute",
        # VM Family
        size="STANDARD_NC6",
        # Minimum running nodes when there is no job running
        min_instances=0,
        # Nodes in cluster
        max_instances=2,
        # How many seconds will the node running after the job termination
        idle_time_before_scale_down=120,
        # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
        tier="Dedicated",
    )

    # Now, we pass the object to MLClient's create_or_update method
    gpu_cluster = ml_client.begin_create_or_update(gpu_cluster).result()

print(
    f"AMLCompute with name {gpu_cluster.name} is created, the compute size is {gpu_cluster.size}"
)

env = ml_client.environments.get(name="keras-env", version="6")

job = command(
    compute=gpu_compute_target,
    environment=f"{env.name}:{env.version}",
    instance_count=1,
    code="./",
    command="python train.py",
    experiment_name="health-chatbot-prediction",
    display_name="classify-977-diseases-create-model-10-epochs",
)

ml_client.jobs.create_or_update(job)
