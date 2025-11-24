"""Convert a Keras / SavedModel to TFLite with optional int8 quantization.

This script attempts to use TensorFlow if available. It supports a simple
representative dataset generator from a directory of images when doing
full int8 quantization. It reports resulting .tflite path, size, and a
single-inference latency measured via the TFLite Interpreter.
"""
from __future__ import annotations
import argparse
import os
import time
import sys
from typing import Iterable

try:
    import tensorflow as tf
    from PIL import Image
    import numpy as np
except Exception:
    tf = None


def representative_dataset_gen(img_dir: str, input_shape):
    files = []
    for e in ('.jpg', '.jpeg', '.png', '.bmp'):
        files.extend([os.path.join(img_dir, p) for p in os.listdir(img_dir) if p.lower().endswith(e)])
    def gen():
        for p in files:
            img = Image.open(p).convert('RGB').resize((input_shape[2], input_shape[1]))
            arr = np.asarray(img).astype(np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)
            yield [arr]
    return gen


def convert(args):
    if tf is None:
        print("TensorFlow is not available in this environment. Conversion cannot proceed.")
        return 1

    if not args.keras_model and not args.saved_model:
        print("Provide either --keras-model or --saved-model")
        return 2

    if args.keras_model:
        model = tf.keras.models.load_model(args.keras_model)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)

    if args.quantize_int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if args.representative_dir and os.path.isdir(args.representative_dir):
            # infer input shape from concrete function if possible
            try:
                if args.keras_model:
                    inp_shape = model.inputs[0].shape.as_list()
                else:
                    # best effort: inspect saved model
                    inp_shape = converter.get_input_details()[0]['shape']
            except Exception:
                inp_shape = [1, 224, 224, 3]
            gen = representative_dataset_gen(args.representative_dir, inp_shape)
            converter.representative_dataset = gen()
        # enforce int8
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    out_path = args.output or (args.keras_model + ".tflite" if args.keras_model else os.path.join(args.saved_model, 'converted.tflite'))
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(out_path) / 1024.0
    print(f"Wrote TFLite model to {out_path} ({size_kb:.1f} KB)")

    # measure single-inference latency if possible
    try:
        import tflite_runtime.interpreter as tflite_rt
        Interpreter = tflite_rt.Interpreter
    except Exception:
        try:
            from tensorflow.lite import Interpreter
        except Exception:
            Interpreter = None
    if Interpreter is not None:
        interp = Interpreter(model_path=out_path)
        interp.allocate_tensors()
        input_details = interp.get_input_details()
        shape = input_details[0]['shape']
        dtype = input_details[0]['dtype']
        dummy = np.zeros(shape, dtype=dtype)
        # warmup
        interp.set_tensor(input_details[0]['index'], dummy)
        interp.invoke()
        runs = 5
        t0 = time.perf_counter()
        for _ in range(runs):
            interp.set_tensor(input_details[0]['index'], dummy)
            interp.invoke()
        t_total = (time.perf_counter() - t0) / runs
        print(f"Single-inference latency (avg over {runs} runs): {t_total*1000.0:.2f} ms")
    else:
        print("TFLite Interpreter not available to measure latency; install tflite-runtime or TensorFlow.")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--keras-model', help='Path to a Keras .h5 or SavedModel dir')
    p.add_argument('--saved-model', help='Path to a SavedModel directory')
    p.add_argument('--output', help='Output .tflite path')
    p.add_argument('--quantize-int8', action='store_true', help='Enable int8 quantization')
    p.add_argument('--representative-dir', help='Directory of images for representative dataset (for int8)')
    args = p.parse_args()
    return_code = convert(args)
    sys.exit(return_code)


if __name__ == '__main__':
    main()
