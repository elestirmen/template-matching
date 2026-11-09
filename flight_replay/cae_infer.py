"""ONNX inference for the cross-autoencoder (no TensorFlow needed).
Input contract (from the training script, recorded in *.pre.json next to the .h5 files):
grayscale -> PIL/TFA-style histogram equalisation -> (x - 127.5) / 127.5 -> model (sigmoid) -> x 255.
"""
import os as _os
import numpy as np
# Persistent CUDA JIT cache (the GPU is newer than CUDA 11.2, so TensorFlow's kernels are JIT-compiled from PTX).
# Reuse the cache that the benchmark already filled, and allow 4 GB so it is not evicted between processes.
_here = _os.path.dirname(_os.path.abspath(__file__))
_bench_cache = _os.path.normpath(_os.path.join(_here, '..', '..', 'benchmark', 'outputs', 'cuda_cache'))
_os.environ.setdefault('CUDA_CACHE_PATH', _bench_cache if _os.path.isdir(_bench_cache) else _os.path.join(_here, 'cuda_cache'))
_os.environ.setdefault('CUDA_CACHE_MAXSIZE', str(4 * 1024 ** 3))
try:
    import onnxruntime as ort
except ImportError:  # the TensorFlow environment does not need onnxruntime
    ort = None


def equalize_pil(gray):
    """Bit-exact re-implementation of PIL ImageOps.equalize / tfa.image.equalize."""
    histogram = np.bincount(gray.reshape(-1), minlength=256).astype(np.int64)
    nonzero = histogram[histogram != 0]
    if nonzero.size <= 1:
        return gray.copy()
    step = (int(nonzero.sum()) - int(nonzero[-1])) // 255
    if step == 0:
        return gray.copy()
    exclusive_cumsum = np.concatenate(([0], np.cumsum(histogram)[:-1]))
    lut = np.clip((exclusive_cumsum + step // 2) // step, 0, 255).astype(np.uint8)
    return lut[gray]


class CAE:
    def __init__(self, onnx_path, threads=0):
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        avail = ort.get_available_providers()
        self.sess = None
        for prov in ('CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider'):
            if prov not in avail: continue
            try:
                if prov == 'DmlExecutionProvider':  # DirectML: no memory patterns, sequential execution
                    so.enable_mem_pattern = False
                    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                self.sess = ort.InferenceSession(onnx_path, so, providers=[prov])
                # a tiny forward pass verifies that the provider really works (CUDA libs may be missing)
                self.sess.run(None, {self.sess.get_inputs()[0].name: np.zeros((1, 1, 544, 544), np.float32)})
                break
            except Exception as e:  # fall back to the next provider
                print(f'[cae_infer] provider {prov} unavailable ({str(e)[:120]}); trying next', flush=True)
                self.sess = None
        if self.sess is None:
            raise RuntimeError('no working onnxruntime provider')
        self.inp = self.sess.get_inputs()[0].name
        self.provider = self.sess.get_providers()[0]
        print(f'[cae_infer] {onnx_path} on {self.provider}', flush=True)

    def run(self, tiles_u8, equalize=True):
        """tiles_u8: list of 544x544 uint8 gray -> list of 544x544 uint8 outputs."""
        batch = []
        for t in tiles_u8:
            g = equalize_pil(t) if equalize else t
            batch.append(((g.astype(np.float32) - 127.5) / 127.5)[None])
        x = np.stack(batch).astype(np.float32)
        y = self.sess.run(None, {self.inp: x})[0]
        return [np.clip(o[0] * 255.0, 0, 255).astype(np.uint8) for o in y]


class SobelTransform:
    """Classical structural representation used as a baseline: gradient magnitude of the
    Gaussian-smoothed (sigma 1.5) grayscale tile, fixed linear scaling (x0.5, clipped to 255).
    Same interface as CAE so that the map and the UAV crops are transformed identically."""
    provider = 'cpu-sobel'

    def run(self, tiles_u8, equalize=True):
        import cv2
        outs = []
        for t in tiles_u8:
            g = equalize_pil(t) if equalize else t
            g = cv2.GaussianBlur(g, (0, 0), 1.5)
            sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(sx * sx + sy * sy) * 0.5
            outs.append(np.clip(mag, 0, 255).astype(np.uint8))
        return outs


def make_transform(name):
    """'' -> None (grayscale), 'sobel' -> SobelTransform, '*.h5' -> KerasCAE (TensorFlow), '*.onnx' -> CAE."""
    if not name: return None
    if name.lower() == 'sobel': return SobelTransform()
    if name.lower().endswith(('.h5', '.hdf5', '.keras')): return KerasCAE(name)
    return CAE(name)


class KerasCAE:
    """Runs the original .h5 model with TensorFlow/Keras (GPU when visible). Same input contract
    and same output conversion as CAE, so the two backends are interchangeable."""

    def __init__(self, h5_path):
        import os
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
        import tensorflow as tf
        self.tf = tf
        gpus = tf.config.list_physical_devices('GPU')
        for g in gpus:
            try: tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError: pass
        try:  # full fp32 (no TF32 on Ampere/Blackwell) so that results do not depend on the GPU generation
            tf.config.experimental.enable_tensor_float_32_execution(False)
        except Exception:
            pass
        from tensorflow.keras.models import load_model

        class _CompatConv2DTranspose(tf.keras.layers.Conv2DTranspose):
            def __init__(self, *args, groups=1, **kw):
                super().__init__(*args, **kw)
        self.model = None; errs = []
        for kw in ({'compile': False}, {'compile': False, 'custom_objects': {'Conv2DTranspose': _CompatConv2DTranspose}}):
            try:
                self.model = load_model(h5_path, **kw); break
            except Exception as e:
                errs.append(str(e)[:200])
        if self.model is None:
            raise RuntimeError('could not load ' + h5_path + ': ' + ' | '.join(errs))
        self.provider = 'tensorflow-' + tf.__version__ + ('-GPU' if gpus else '-CPU')
        print(f'[cae_infer] {h5_path} on {self.provider}', flush=True)

    def run(self, tiles_u8, equalize=True):
        batch = []
        for t in tiles_u8:
            g = equalize_pil(t) if equalize else t
            batch.append(((g.astype(np.float32) - 127.5) / 127.5)[..., None])
        x = np.stack(batch).astype(np.float32)
        y = self.model(x, training=False).numpy()
        return [np.clip(o[..., 0] * 255.0, 0, 255).astype(np.uint8) for o in y]
