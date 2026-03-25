import yaml
import argparse
import logging
import os
from src.data.processor import DataProcessor
from src.models.model_builder import build_optimization_model
from src.training.trainer import Trainer


def setup_logging(log_dir: str) -> None:
    """Sets up logging to both console and file for traceability."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log")),
            logging.StreamHandler(),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transportation Optimization AI Training Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found.")
        return

    setup_logging(config["paths"]["log_dir"])
    logger = logging.getLogger(__name__)
    os.makedirs(config["paths"]["checkpoint_dir"], exist_ok=True)

    logger.info("Starting Transportation Optimization AI Pipeline")

    # --- Data Loading & Processing ---
    logger.info("Processing data...")
    processor = DataProcessor(config["data"])
    train_ds, val_ds = processor.process_pipeline(config["data"]["file_path"])

    # --- Model Construction ---
    logger.info("Building model from config...")
    model = build_optimization_model(config["model"])
    model.summary(print_fn=logger.info)

    # --- Training ---
    logger.info("Beginning training...")
    trainer = Trainer(model, config)
    trainer.train(train_ds, val_ds)

    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
