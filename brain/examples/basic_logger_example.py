"""Basic example demonstrating DataLogger usage."""
import logging
import time

from config.config_loader import ConfigLoader
from data_collection import DataLogger


def main():
    logging.basicConfig(level=logging.INFO)
    cfg = ConfigLoader()
    dl = DataLogger(cfg)
    dl.start()
    try:
        time.sleep(10)
        print("Status:", dl.get_status())
    finally:
        dl.stop()


if __name__ == "__main__":
    main()
