import yaml
import os
import signal
from apscheduler.schedulers.blocking import BlockingScheduler
from xinference_client import RESTfulClient as Client
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger


load_dotenv(override=True)

XINFERENCE_URL = os.getenv("XINFERENCE_URL")
CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", 300))

# Validate environment variables first
if not XINFERENCE_URL:
    raise ValueError("XINFERENCE_URL environment variable is not set.")

# Initialize client after validation
CLIENT = Client(XINFERENCE_URL)
scheduler = BlockingScheduler()

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    logger.info(f"Received signal {signum} ({signal.strsignal(signum)}), shutting down gracefully...")
    try:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler shutdown complete")
    except Exception as e:
        logger.warning(f"Error during scheduler shutdown: {e}")
    finally:
        logger.info("Exiting process")
        os._exit(0)  # Force exit without cleanup

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# todo save logs to file
# ...

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
    try:
        logger.info("Starting xinference daemon...")
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Received shutdown signal, stopping scheduler...")
        try:
            scheduler.shutdown(wait=False)
        except Exception as e:
            logger.warning(f"Error during scheduler shutdown: {e}")
        logger.info("Daemon stopped successfully")
        os._exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        try:
            scheduler.shutdown(wait=False)
        except Exception:
            pass
        os._exit(1)
