"""Example demonstrating model comparison workflow.
"""
from config.config_loader import ConfigLoader
from scripts.compare_models import main as compare_main

def main():
    # This example wraps the CLI compare script; run via: python -m examples.model_comparison_example
    compare_main()

if __name__ == '__main__':
    main()
