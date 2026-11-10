"""Model registry shared by run_all.py and texture_probe.py.

Backend: 'keras' runs the original .h5 files with TensorFlow (GPU in the visual_navigation_cuda
environment); 'onnx' runs the converted .onnx files with onnxruntime. Choose with the environment
variable VN_BACKEND; by default 'keras' is used when TensorFlow can be imported.
Paths are relative to the template-matching folder.
"""
import os

H5 = {
    'm4e300': 'top_modeller/GPU_model_f48_k5_epoch_00300_sigmoid_(2, 2)_.h5',            # SHA-256 d2228607b6538784...
    'f48k4s1': '../benchmark/models/03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__cb62bfb86ad0.h5',  # ckpt 1850
    'f48k4s1c1550': 'model/_03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__bdb7daa618c0.h5',          # ckpt 1550
}
# The same three files are public in https://github.com/elestirmen/benchmark (verified byte-identical, 10 Nov 2026):
PUBLIC = {  # label: (path in the benchmark repository, SHA-256)
    'm4e300': ('diger_modeller/kosmus_GPU_modef_48_k5_ELU_S2/GPU_model_f48_k5_epoch_00300_sigmoid_(2, 2)_.h5',
               'd2228607b653878469455dd0e27f38ef936687d4df5cb772975e233ff52a52e1'),
    'f48k4s1': ('top_modeller/_03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__cb62bfb86ad0.h5',
                'cb62bfb86ad05abb0e29ce0a84ae4b98916953f1e6ac0e93967e6136603c0229'),
    'f48k4s1c1550': ('top_modeller/_03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__bdb7daa618c0.h5',
                     'bdb7daa618c04ac7e6913ff6aaa88010be41bda7920d4505a470cf7fe5e75f68'),
}
ONNX = {
    'm4e300': 'flight_replay/onnx/f48k5s2_e300.onnx',
    'f48k4s1': 'flight_replay/onnx/f48k4s1_ckpt1850_cb62.onnx',
    'f48k4s1c1550': 'flight_replay/onnx/f48k4s1_ckpt1550_bdb7.onnx',
}


def backend():
    b = os.environ.get('VN_BACKEND', '').strip().lower()
    if b in ('keras', 'onnx'):
        return b
    try:
        import importlib.util
        return 'keras' if importlib.util.find_spec('tensorflow') else 'onnx'
    except Exception:
        return 'onnx'


def model_path(label):
    return (H5 if backend() == 'keras' else ONNX)[label]
