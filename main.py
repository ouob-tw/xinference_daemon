from pprint import pprint
import yaml
import os
from pathlib import Path
from xinference_client import RESTfulClient as Client
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

XINFERENCE_URL = os.getenv("XINFERENCE_URL")

if not XINFERENCE_URL:
    raise ValueError("XINFERENCE_URL environment variable is not set.")
if not Path("config.yaml").exists():
    raise FileNotFoundError(
        "config.yaml file is missing. Please create it with the necessary configuration."
    )

with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)
    if "models" not in config:
        raise ValueError("Invalid configuration in config.yaml.")

client = Client(XINFERENCE_URL)
running_models = client.list_models()

for d in config.get("models"):
    model_name = d.get("name")
    model_type = d.get("type")
    model_engine = d.get("engine")
    model_uid = d.get("uid")

    if model_uid in running_models:
        continue

    model_uid = client.launch_model(
        model_name=model_name, model_type=model_type, model_uid=model_uid
    )
    logger.info(f'Model "{model_name}" launched with UID: {model_uid}')
