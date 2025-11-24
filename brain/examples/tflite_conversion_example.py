"""Example wrapper to run the TFLite conversion script.

This example demonstrates how to call the conversion module programmatically.
"""
from scripts.convert_mediapipe_to_tflite import convert
import argparse

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--keras-model')
    p.add_argument('--output')
    p.add_argument('--quantize-int8', action='store_true')
    p.add_argument('--representative-dir')
    args = p.parse_args()
    class Args:
        keras_model = args.keras_model
        saved_model = None
        output = args.output
        quantize_int8 = args.quantize_int8
        representative_dir = args.representative_dir
    convert(Args())
