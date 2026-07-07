import logging
import os

from transformers import CLIPModel, CLIPProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "openai/clip-vit-base-patch32"

SAVE_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "local_models", "clip")


def download_and_save():
    logger.info(f"Starting download of {MODEL_NAME}...")

    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    os.makedirs(SAVE_DIRECTORY, exist_ok=True)

    logger.info(f"Saving in {SAVE_DIRECTORY}...")
    model.save_pretrained(SAVE_DIRECTORY)
    processor.save_pretrained(SAVE_DIRECTORY)

    logger.info("Completed")


if __name__ == "__main__":
    download_and_save()
