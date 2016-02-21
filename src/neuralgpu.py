import mxnet as mx
import numpy as np
import math, sys

class AbsValueLayer(mx.operator.NumpyOp):
    def __init__(self):
        super(AbsValueLayer, self).__init__(True)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        y[:] = np.absolute(x)

    def backward(self, out_grad, in_data, out_data, in_grad):
        x = in_data[0]
        dx = in_grad[0]
        dx[:] = out_grad * np.sign(x)


class TransformInputAcrossMaps:
    @staticmethod
    def transform_input(ctx, data, batch_size, domain_size, width, input_size, num_maps, out_extra_shapes):
        """ Each input value (expected as one-hot) is embedded such that input_maps[0,i,:] = embedding(input[i])"""
        embedding_w = mx.sym.Variable("embedding_weight")

        padding = mx.sym.Variable("padding")

        out_extra_shapes['embedding_weight'] = (domain_size, num_maps)
        out_extra_shapes["input"] = (batch_size, input_size)
        out_extra_shapes["padding"] = (batch_size, width - 1, input_size, num_maps)

        # ok, this is a hack, but I don't know how to do an embedding if my input is three dimensional
        data = mx.symbol.Reshape(data=data, target_shape=(batch_size * input_size,), name='embedding_reshape1')
        data = mx.symbol.Embedding(data=data, weight=embedding_w, input_dim=domain_size, output_dim=num_maps, name='embedding')

        data = mx.symbol.Reshape(data=data, target_shape=(batch_size, 1, input_size, num_maps), name='embedding_reshape2')
        # this requires bleeding edge mxnet to work. see https://github.com/dmlc/mxnet/issues/1130
        data = mx.symbol.Concat(data, padding, num_args=2, dim=1, name='embedding_concat')
        data = mx.symbol.SwapAxis(data=data, dim1=1, dim2=3, name="embedding_swap")
        return data
        

    @staticmethod
    def transform_output(ctx, data, label, batch_size, domain_size, width, input_size, num_maps, out_extra_shapes):
        output_w = mx.sym.Variable("output_weight")

        out_extra_shapes['output_weight'] = (domain_size, num_maps)
        #out_extra_shapes['label'] = (batch_size, input_size, domain_size)

        data = mx.symbol.SwapAxis(data=data, dim1=1, dim2=3, name="output_swap1") # now the shape is (batch_size, w, input_size, num_maps)
        data = mx.symbol.SwapAxis(data=data, dim1=1, dim2=2, name="output_swap2") # now the shape is (batch_size, input_size, w, num_maps)
        data = mx.symbol.Reshape(data=data, target_shape=(batch_size * input_size, num_maps * width), name='output_reshape1')
        splits = mx.symbol.SliceChannel(data=data, name='output_split', num_outputs=width)

        data = splits[0]

        # mxnet requires all the dangling outputs to be passed to `bind`, so block gradient from the remainder of the last layer
        block = []
        for i in range(1, width):
            block.append(mx.symbol.BlockGrad(data=splits[i]))

        data = mx.symbol.FullyConnected(data=data, weight=output_w, no_bias=True, num_hidden=domain_size, name='output_transform')

        data = mx.symbol.Reshape(data=data, target_shape=(batch_size, input_size, domain_size), name='output_reshape2')

               

        # in mxnet SoftmaxOutput expects 1-st (0-based) dimension to be classes. Shuffling to achieve it
        data = mx.symbol.SwapAxis(data=data, dim1=1, dim2=2, name="output_swap3")
        data = mx.symbol.SoftmaxOutput(data=data, multi_output=True, label=label, name="output")
        #data = SoftmaxWithRelaxation()(data=data, label=label, name="output")

        return mx.symbol.Group([data] + block)
        



class NeuralGPU:
    # set of helper functions to build the neural GPU
    def _build_gated_conv_unit(self, data, label, prefix, kernel_size, num_maps, num_layers, timestep):
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1, "Kernel for the convolutional gated unit must have odd dimension sizes"
        par = timestep % self._r
        pad_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        for layer_n in range(num_layers):
            u = mx.sym.Convolution(data=data, name='conv_' + prefix + 'u_%d' % layer_n, weight=self._u_w[par][layer_n], bias=self._u_b[par][layer_n], kernel=kernel_size, pad=pad_size, num_filter=num_maps)
            u = mx.sym.Activation(data=u, act_type='sigmoid', name='act_' + prefix + 'u_%d' % layer_n)
            r = mx.sym.Convolution(data=data, name='conv_' + prefix + 'r_%d' % layer_n, weight=self._r_w[par][layer_n], bias=self._r_b[par][layer_n], kernel=kernel_size, pad=pad_size, num_filter=num_maps)
            r = mx.sym.Activation(data=r, act_type='sigmoid', name='act_' + prefix + 'r_%d' % layer_n)

            t = mx.sym.Convolution(data=(r * data), name='conv_' + prefix + 't_%d' % layer_n, weight=self._t_w[par][layer_n], bias=self._t_b[par][layer_n], kernel=kernel_size, pad=pad_size, num_filter=num_maps)
            t = mx.sym.Activation(data=t, act_type='tanh', name='act_' + prefix + 't_%d' % layer_n)

            data = u * data + (1 - u) * t
            data = mx.sym.Dropout(data=data, p=0.07)

        return data


    def _build_relaxation_cost_single(self, weights):
        s = weights[0]
        for weight in weights[1:]:
            s += weight

        s /= float(len(weights))
        #ret = AbsValueLayer()(weights[0] - s)
        ret = (weights[0] - s) * (weights[0] - s)
        for weight in weights[1:]:
            #ret += AbsValueLayer()(data=weight - s)
            ret += (weight - s) * (weight - s)

        return ret


    def _build_relaxation_cost(self, relaxation_pull):
        ret_w = None
        ret_b = None
        for w in [self._u_w, self._r_w, self._t_w]:
            for layer_n in range(self._num_layers):
                addend = self._build_relaxation_cost_single([w[x][layer_n] for x in range(self._r)])
                if ret_w is None:
                    ret_w = addend
                else:
                    ret_w += addend
        for w in [self._u_b, self._r_b, self._t_b]:
            for layer_n in range(self._num_layers):
                addend = self._build_relaxation_cost_single([w[x][layer_n] for x in range(self._r)])
                if ret_b is None:
                    ret_b = addend
                else:
                    ret_b += addend
        ret_w *= relaxation_pull
        ret_b *= relaxation_pull
        # using AbsValueLayer here as a Identity to add name (values are always positive at this moment)
        #ret_w = AbsValueLayer()(data=ret_w, name='relax0')
        #ret_b = AbsValueLayer()(data=ret_b, name='relax1')
        return (ret_w, ret_b)


    def _build_neural_gpu(self, data, label, batch_size, input_size, width, depth, relaxation_pull):
        data = self._input_transformer.transform_input(self._ctx, data, batch_size, self._domain_size, width, input_size, self._num_maps, self._shapes)

        for i in range(depth):
            data = self._build_gated_conv_unit(data, label, "s%d_" % i, self._kernel_size, self._num_maps, self._num_layers, i)

        data = self._input_transformer.transform_output(self._ctx, data, label, batch_size, self._domain_size, width, input_size, self._num_maps, self._shapes)

        return mx.sym.Group((data,) + self._build_relaxation_cost(relaxation_pull))
        

    def __init__(self, domain_size, input_transformer, num_layers=2, kernel_size=(3, 3), num_maps=24):
        self._domain_size = domain_size
        self._kernel_size = kernel_size
        self._num_maps = num_maps
        self._num_layers = num_layers
        self._input_transformer = input_transformer

        self._stored_weights = None
        self._updater = None


    def is_param_name(self, name):
        return name.endswith("_weight") or name.endswith("_bias") or name.endswith("gamma") or name.endswith("beta")


    @staticmethod
    def get_batch(X, fr, l):
        if fr + l <= X.shape[0]:
            return X[fr:fr+l]
        else:
            a = X[fr:]
            b = X[:l - (X.shape[0] - fr)]
            return np.concatenate((a, b))


    def _average_relaxed_parameter_single(self, param, param_suffix, layer_no):
        result = self._stored_weights["%s_%s_%s_%s" % (param, layer_no, 0, param_suffix)]
        for par in range(1, self._r):
            result = self._stored_weights["%s_%s_%s_%s" % (param, layer_no, par, param_suffix)]

        result /= float(self._r)

        self._stored_weights["%s_%s_%s_%s" % (param, layer_no, 0, param_suffix)] = result


    def average_relaxed_parameters(self):
        for layer_no in range(self._num_layers):
            self._average_relaxed_parameter_single("u_w", "weight", layer_no)
            self._average_relaxed_parameter_single("r_w", "weight", layer_no)
            self._average_relaxed_parameter_single("t_w", "weight", layer_no)
            self._average_relaxed_parameter_single("u_b", "bias", layer_no)
            self._average_relaxed_parameter_single("r_b", "bias", layer_no)
            self._average_relaxed_parameter_single("t_b", "bias", layer_no)

        self._r = 1


    # TODO: make r=6
    def train(self, ctx, input_size, width, depth, X, y, batch_size, num_epochs, learning_rate, momentum, initializer, relaxation_pull, X_val=None, y_val=None, optimizer="adam", max_grad_norm=5, r=6, verbose=True, **kwargs):
        self._ctx = ctx

        self._shapes = {}
        self._r = r
        self._u_w, self._r_w, self._t_w, self._u_b, self._r_b, self._t_b = [[[] for i in range(r)] for j in range(6)]
        # r is number of shared parameters, see "parameter sharing relaxation" section of the paper
        for par in range(r):
            for layer_n in range(self._num_layers):
                self._u_w[par].append(mx.sym.Variable("u_w_%d_%d_weight" % (layer_n, par)))
                self._r_w[par].append(mx.sym.Variable("r_w_%d_%d_weight" % (layer_n, par)))
                self._t_w[par].append(mx.sym.Variable("t_w_%d_%d_weight" % (layer_n, par)))
                self._u_b[par].append(mx.sym.Variable("u_b_%d_%d_bias" % (layer_n, par)))
                self._r_b[par].append(mx.sym.Variable("r_b_%d_%d_bias" % (layer_n, par)))
                self._t_b[par].append(mx.sym.Variable("t_b_%d_%d_bias" % (layer_n, par)))

        X = np.array(X)
        y = np.array(y)

        if X_val is not None:
            assert y_val is not None
            X_val = np.array(X_val)
            y_val = np.array(y_val)
            #X_val = (X_val - self._X_mean) / self._X_std
        else:
            assert y_val is None

        if optimizer == 'adam' and 'epsilon' not in kwargs:
            kwargs['epsilon'] = 1e-5 # as per the paper

        if not self._updater:
            optimizer = mx.optimizer.create(optimizer, learning_rate=learning_rate, **kwargs)
            self._optimizer = optimizer
            self._updater = mx.optimizer.get_updater(optimizer)

        self._optimizer.wd = 0.1 * math.sqrt(input_size)

        inp = mx.sym.Variable('input')
        label = mx.sym.Variable('label')
        model_sym = self._build_neural_gpu(inp, label, batch_size, input_size, width, depth, relaxation_pull)

        arg_names = model_sym.list_arguments()

        # most of the code here is from the LSTM tutorial
        arg_shape, out_shape, aux_shape = model_sym.infer_shape(**self._shapes)
        args_grad = {}
        assert (initializer is None) == (self._stored_weights is not None)
        arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape] if initializer is not None else [mx.nd.array(self._stored_weights[n], ctx) if self.is_param_name(n) else mx.nd.zeros(s, ctx) for (n, s) in zip(arg_names, arg_shape)]

        for shape, name in zip(arg_shape, arg_names):
            if self.is_param_name(name):
                args_grad[name] = mx.nd.zeros(shape, ctx)

        model_exec = model_sym.bind(ctx=self._ctx, args=arg_arrays,
                                args_grad=args_grad,
                                grad_req="add")


        out_dict = dict(zip(model_sym.list_outputs(), model_exec.outputs))
        arg_dict = dict(zip(arg_names, arg_arrays))

        param_blocks = []
        for i, name in enumerate(arg_names):
            if self.is_param_name(name):
                if initializer is not None:
                    initializer(name, arg_dict[name])

                param_blocks.append((i, arg_dict[name], args_grad[name], name))
            else:
                assert name not in args_grad

        output = out_dict["output_output"]

        inp = arg_dict["input"]
        label = arg_dict["label"]

        beginning = 0
        num_samples = len(X)
        accum_val_acc = 0
        for epoch_n in range(num_epochs):
            if verbose:
                print " | Epoch %3d ========" % epoch_n,
            for it in range((num_samples + batch_size - 1) // batch_size):
                mx.nd.array(NeuralGPU.get_batch(X, beginning, batch_size)).copyto(inp)
                mx.nd.array(NeuralGPU.get_batch(y, beginning, batch_size)).copyto(label)

                grads = [mx.nd.empty(shape, self._ctx) for shape in out_shape]

                model_exec.forward(is_train=True)

                # seems like one can't selectively provide gradient only for some outputs, so provide it for all of them
                # SoftmaxOutput appears to expect the actual output as the gradient (since it subtracts labels from it, and the gradient of softmax+logloss is out-label)
                for i, name in enumerate(model_sym.list_outputs()):
                    out_dict[name].copyto(grads[i])

                model_exec.backward(grads)

                # updare parameters
                norm = 0.
                for idx, weight, grad, name in param_blocks:
                    grad /= (batch_size * input_size)
                    l2_norm = mx.nd.norm(grad).asscalar()
                    norm += l2_norm*l2_norm
                norm = math.sqrt(norm)
                if norm > max_grad_norm:
                    if norm > max_grad_norm * 10:
                        print "!",
                    else:
                        print ".",
                for idx, weight, grad, name in param_blocks:
                    if norm > max_grad_norm:
                        grad *= (max_grad_norm / norm)
                    self._updater(idx, grad, weight)
                    # reset gradient to zero
                    grad[:] = 0.0

                beginning += batch_size
                beginning %= num_samples

            if X_val is not None:
                mx.nd.array(NeuralGPU.get_batch(X_val, 0, batch_size)).copyto(inp)
                y_val = y_val[:batch_size]
                model_exec.forward(is_train=False)
                out_np = np.argmax(output.asnumpy(), axis=1)

                a = np.sum(out_np == y_val)
                b = y_val.shape[0] * y_val.shape[1]
                print "%5d/%5d = %.10lf" % (a, b, 1.0 * a / b),
                sys.stdout.flush()
                accum_val_acc += 1.0 * a / b

        self._stored_weights = {}
        for k, v in arg_dict.items():
            v.wait_to_read()
            self._stored_weights[k] = v.asnumpy()

        print ""

        return accum_val_acc / num_epochs

# returns the performance
def reset_at(task):
    if task == 'add': return 0.82
    else: return 0.70

def gen_samples(task, cur_len):
    def to_bin(a, out, l, onehot=False):
        for i in range(l):
            bit = a % 2
            if onehot: out.append([bit, 1 - bit, 0])
            else: out.append(bit)
            a //= 2

    X = []
    y = []
    X_val = []
    y_val = []
    total_samples = 512
    test_samples = 200

    for i in range(total_samples):
        inp = []
        out = []
        if task == 'mul' or task == 'add':
            a = random.randint(1, 1 << cur_len)
            b = random.randint(1, 1 << cur_len)
            c = a * b if task == 'mul' else a + b
            to_bin(a, inp, cur_len)
            inp.append(2)
            to_bin(b, inp, cur_len)
            to_bin(c, out, cur_len + cur_len + 1)

        elif task in ['copy', 'reverse', 'invert']:
            a = random.randint(1, 1 << (cur_len * 2 + 1))
            to_bin(a, inp, cur_len * 2 + 1)
            to_bin(a, out, cur_len * 2 + 1)
            if task == 'reverse':
                out = list(reversed(out))
            elif task == 'invert':
                out = [1 - x for x in out]

        else: assert False, "Unknown task type %s" % task

        if i < total_samples - test_samples:
            X.append(inp)
            y.append(out)
        else:
            X_val.append(inp)
            y_val.append(out)

    return X, y, X_val, y_val


import random
if __name__ == "__main__":
    task = 'mul'
    if len(sys.argv) > 1:
        task = sys.argv[1]

    ng = NeuralGPU(3, TransformInputAcrossMaps)

    relaxation_pull = 0.0005
    curric_len = 2
    first = True
    iter_ord = 0
    bad_iters = 0

    history = []

    while True:
        iter_ord += 1
        cur_len = curric_len

        if curric_len >= 3:
            if iter_ord % 6 == 5:
                cur_len = random.randint(3, 20)
            elif iter_ord % 2 == 1:
                cur_len = random.randint(3, curric_len)

        print "CUR LEN ", cur_len, " ... ",
        num_epochs = 2 if cur_len == curric_len else 1
        sys.stdout.flush()
        X, y, X_val, y_val = gen_samples(task, cur_len)
        ctx = mx.gpu(0)
        try:
            val = ng.train(ctx, cur_len + cur_len + 1, 4, cur_len + cur_len, X, y, 64, num_epochs, 0.01, 0.9, mx.initializer.Uniform(0.3) if first else None, relaxation_pull, X_val=X_val, y_val=y_val)
        except Exception as e:
            # sometimes memory errors happen, but seem to recover by the next iteration
            print e

        history.append(ng._stored_weights)
        if len(history) > 20:
            history = history[1:]
            assert len(history) == 20

        first = False

        if val >= 0.998 and cur_len == curric_len:
            relaxation_pull *= 1.2
            curric_len += (cur_len + 19) // 20 # +1 for the first 20 iters, then more and more
            iter_ord = 0

        if val < reset_at(task) and cur_len == curric_len:
            bad_iters += 1
        elif cur_len == curric_len:
            bad_iters = 0

        if bad_iters >= 10:
            print "ROLLING THE MODEL 20 ITERATIONS BACK"
            bad_iters = 0
            ng._stored_weights = history[0]
            history = history[0:1]

        if curric_len == 200:
            break

