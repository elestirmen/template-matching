"""Checks the TensorFlow backend on this machine: GPU visibility, agreement with the reference
outputs computed with the ONNX conversion (ref/cae_reference.npz), and speed."""
import os, sys, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf
print('TensorFlow', tf.__version__, '| GPUs:', [g.name for g in tf.config.list_physical_devices('GPU')])
import models_registry as MR
from cae_infer import KerasCAE
ref = np.load(os.path.join(HERE, 'ref', 'cae_reference.npz'))
tile = ref['tile']
ok = True
for label in ('m4e300', 'f48k4s1', 'f48k4s1c1550'):
    m = KerasCAE(os.path.join(ROOT, MR.H5[label]))
    out = m.run([tile])[0].astype(np.int16)
    d = np.abs(out - ref[label].astype(np.int16))
    print(f'{label}: max |diff| = {d.max()} gray levels, mean = {d.mean():.3f}, pixels differing > 1: {(d > 1).mean() * 100:.3f}%')
    if d.max() > 3 or (d > 1).mean() > 0.001: ok = False
    m.run([tile] * 8)
    t = time.time()
    for _ in range(5): m.run([tile] * 8)
    print(f'   speed: {40 / (time.time() - t):.1f} tiles/s')
print('KERAS CHECK:', 'OK' if ok else 'MISMATCH')
sys.exit(0 if ok else 3)
