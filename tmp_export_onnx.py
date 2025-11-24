from ultralytics import YOLO
import shutil, glob, os
print('Exporting yolov8n -> ONNX (opset=12)...')
model = YOLO('yolov8n.pt')   # downloads weights if missing
model.export(format='onnx', opset=12, imgsz=320)
c = glob.glob('**/*.onnx', recursive=True)
print('Found ONNX files (last 3):', c[-3:])
out = r'brain\\models\\yolo_face_onnx.onnx'
os.makedirs(os.path.dirname(out), exist_ok=True)
if c:
    shutil.copy2(c[-1], out)
    print('Copied', c[-1], '->', out)
else:
    print('No ONNX found; check export output')
