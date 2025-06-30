import yaml
import os
from apscheduler.schedulers.blocking import BlockingScheduler
from xinference_client import RESTfulClient as Client
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger


load_dotenv(override=True)

XINFERENCE_URL = os.getenv("XINFERENCE_URL")
CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", 300))
CLIENT = Client(XINFERENCE_URL)
scheduler = BlockingScheduler()

# todo save logs to file

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


def models_daemon():
    running_models = CLIENT.list_models()

    for d in config.get("models"):
        model_name = d.get("name")
        model_type = d.get("type")
        model_engine = d.get("engine")
        model_uid = d.get("uid")

        if model_uid in running_models:
            continue

        model_uid = CLIENT.launch_model(
            model_name=model_name,
            model_type=model_type,
            model_uid=model_uid,
            model_engine=model_engine,
        )
        logger.warning(
            f'Model "{model_name}" NOT RUNNING, launch now with UID: {model_uid}'
        )


@scheduler.scheduled_job("interval", seconds=CHECK_INTERVAL_SECONDS)
def job1():
    logger.info("Running models daemon job...")
    models_daemon()


if __name__ == "__main__":
    scheduler.start()
