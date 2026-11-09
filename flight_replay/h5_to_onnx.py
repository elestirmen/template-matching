"""Keras (.h5, TF2) symmetric conv encoder-decoder -> ONNX, without TensorFlow.
Supports Conv2D / Conv2DTranspose with 'same' padding, ELU/ReLU/sigmoid/tanh/linear.
"""
import sys, json, numpy as np, h5py, onnx
from onnx import helper, TensorProto, numpy_helper

def tf_same_pads(in_size, k, s):
    out = -(-in_size // s)
    total = max((out - 1) * s + k - in_size, 0)
    before = total // 2
    return before, total - before, out

def convert(h5_path, onnx_path, in_size=544):
    h = h5py.File(h5_path, 'r')
    cfg = h.attrs['model_config']
    if isinstance(cfg, bytes): cfg = cfg.decode()
    cfg = json.loads(cfg)
    layers = cfg['config']['layers']
    wroot = h['model_weights'] if 'model_weights' in h else h
    nodes, inits = [], []
    cur = 'input'; size = in_size
    cur_c = 1
    acts = {'elu': 'Elu', 'relu': 'Relu', 'sigmoid': 'Sigmoid', 'tanh': 'Tanh', 'linear': None}
    li = 0
    for L in layers:
        cn = L['class_name']; lc = L['config']; name = lc['name']
        if cn not in ('Conv2D', 'Conv2DTranspose'):
            if cn in ('InputLayer', 'Functional', 'Sequential'): continue
            raise ValueError('unsupported layer ' + cn)
        g = wroot[name]
        # find kernel/bias datasets (nested group name)
        ds = {}
        def collect(n, o):
            if isinstance(o, h5py.Dataset): ds[n.split('/')[-1]] = o[()]
        g.visititems(collect)
        kern = [v for k, v in ds.items() if 'kernel' in k][0]
        bias = [v for k, v in ds.items() if 'bias' in k]
        bias = bias[0] if bias else None
        k = lc['kernel_size'][0]; s = lc['strides'][0]
        assert lc['padding'] == 'same', lc['padding']
        filters = lc['filters']
        li += 1
        wname = f'W{li}'; bname = f'B{li}'; out = f'y{li}'
        if cn == 'Conv2D':
            W = np.transpose(kern, (3, 2, 0, 1)).astype(np.float32)  # HWIO -> OIHW
            pb, pa, osz = tf_same_pads(size, k, s)
            node = helper.make_node('Conv', [cur, wname] + ([bname] if bias is not None else []), [out],
                                    kernel_shape=[k, k], strides=[s, s], pads=[pb, pb, pa, pa])
            size = osz
        else:
            W = np.transpose(kern, (3, 2, 0, 1)).astype(np.float32)  # HW O I -> I O H W
            osz = size * s
            # TF same transpose: full=(in-1)*s+k ; crop before=pb, after=pa with pb+pa = full-osz
            total = (size - 1) * s + k - osz
            pb = total // 2; pa = total - pb
            node = helper.make_node('ConvTranspose', [cur, wname] + ([bname] if bias is not None else []), [out],
                                    kernel_shape=[k, k], strides=[s, s], pads=[pb, pb, pa, pa])
            size = osz
        inits.append(numpy_helper.from_array(W, wname))
        if bias is not None:
            inits.append(numpy_helper.from_array(bias.astype(np.float32), bname))
        nodes.append(node)
        cur = out
        a = acts[lc['activation']]
        if a:
            nodes.append(helper.make_node(a, [cur], [cur + 'a']))
            cur = cur + 'a'
        cur_c = filters
        print(f'{name}: {cn} f={filters} k={k} s={s} act={lc["activation"]} -> {size}x{size}x{filters}')
    inp = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, 1, in_size, in_size])
    outp = helper.make_tensor_value_info(cur, TensorProto.FLOAT, [None, cur_c, size, size])
    graph = helper.make_graph(nodes, 'cae', [inp], [outp], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)
    nparams = sum(int(np.prod(t.dims)) for t in inits)
    print('saved', onnx_path, 'params', nparams, 'output', cur)
    return cur

if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2])
