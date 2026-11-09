"""Quick check: which onnxruntime provider is used and how fast the CAE runs."""
import os, sys, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import onnxruntime as ort
print('onnxruntime', ort.__version__, ort.get_available_providers())
from cae_infer import CAE
c = CAE(os.path.join(HERE, 'onnx', 'f48k4s1_ckpt1850_cb62.onnx'))
x = [np.random.default_rng(0).integers(0, 255, (544, 544), dtype=np.uint8)] * 8
c.run(x)
t = time.time()
for _ in range(5): c.run(x)
dt = time.time() - t
print(f'provider: {c.provider} | 40 tiles in {dt:.1f} s ({40 / dt:.1f} tiles/s)')
