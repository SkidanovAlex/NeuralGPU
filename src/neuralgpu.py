import mxnet as mx
import numpy as np
import math


class TransformInputAcrossMaps:
    @staticmethod
    def transform_input(ctx, data, batch_size, domain_size, width, input_size, num_maps, out_extra_shapes):
        """ Each input value (expected as one-hot) is embedded such that input_maps[0,i,:] = embedding(input[i])"""
        embedding_w = mx.sym.Variable("embedding_weight")
        padding = mx.sym.Variable("padding")

        out_extra_shapes['embedding_weight'] = (domain_size, num_maps)
        out_extra_shapes["input"] = (batch_size, input_size)
        #out_extra_shapes["padding"] = (batch_size, width - 1, input_size, num_maps)

        # ok, this is a hack, but I don't know how to do an embedding if my input is three dimensional
        data = mx.symbol.Reshape(data=data, target_shape=(batch_size * input_size,), name='embedding_reshape1')
        data = mx.symbol.Embedding(data=data, weight=embedding_w, input_dim=domain_size, output_dim=num_maps, name='embedding')

        padding = mx.sym.BlockGrad(data=padding, name="padding_block")

        data = mx.symbol.Reshape(data=data, target_shape=(batch_size, 1, input_size, num_maps), name='embedding_reshape2')
        data = mx.symbol.Concat(data, data, data, data, num_args=4, dim=1, name='embedding_concat')
        #data = mx.symbol.Concat(data, padding, num_args=2, dim=1, name='embedding_concat')
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
        for i in range(1, 4):
            block.append(mx.symbol.BlockGrad(data=splits[i]))

        data = mx.symbol.FullyConnected(data=data, weight=output_w, no_bias=True, num_hidden=domain_size, name='output_transform')

        data = mx.symbol.Reshape(data=data, target_shape=(batch_size, input_size, domain_size), name='output_reshape2')

        # in mxnet SoftmaxOutput expects 1-st (0-based) dimension to be classes. Shuffling to achieve it
        data = mx.symbol.SwapAxis(data=data, dim1=1, dim2=2, name="output_swap3")
        data = mx.symbol.SoftmaxOutput(data=data, multi_output=True, label=label, name="output")

        return mx.symbol.Group([data] + block)
        



class NeuralGPU:
    # set of helper functions to build the neural GPU
    def _build_gated_conv_unit(self, data, label, prefix, kernel_size, num_maps, num_layers):
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1, "Kernel for the convolutional gated unit must have odd dimension sizes"
        pad_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        for layer_n in range(num_layers):
            u = mx.sym.Convolution(data=data, name=prefix + 'u_%d' % layer_n, weight=self._u_w[layer_n], bias=self._u_b[layer_n], kernel=kernel_size, pad=pad_size, num_filter=num_maps)
            u = mx.sym.Activation(data=u, act_type='sigmoid')
            r = mx.sym.Convolution(data=data, name=prefix + 'r_%d' % layer_n, weight=self._r_w[layer_n], bias=self._r_b[layer_n], kernel=kernel_size, pad=pad_size, num_filter=num_maps)
            r = mx.sym.Activation(data=r, act_type='sigmoid')

            t = mx.sym.Convolution(data=(r * data), name=prefix + 't_%d' % layer_n, weight=self._t_w[layer_n], bias=self._t_b[layer_n], kernel=kernel_size, pad=pad_size, num_filter=num_maps)
            t = mx.sym.Activation(data=t, act_type='tanh')

            data = u * data + (1 - u) * t

        return data


    def _build_neural_gpu(self, data, label, batch_size, input_size, width, depth):
        data = self._input_transformer.transform_input(self._ctx, data, batch_size, self._domain_size, width, input_size, self._num_maps, self._shapes)

        for i in range(depth):
            data = self._build_gated_conv_unit(data, label, "s%d_" % i, self._kernel_size, self._num_maps, self._num_layers)

        data = self._input_transformer.transform_output(self._ctx, data, label, batch_size, self._domain_size, width, input_size, self._num_maps, self._shapes)

        return data
        

    def __init__(self, ctx, domain_size, input_transformer, num_layers=2, kernel_size=(3, 3), num_maps=24):
        self._shapes = {}
        self._u_w, self._r_w, self._t_w, self._u_b, self._r_b, self._t_b = [[] for i in range(6)]
        for layer_n in range(num_layers):
            self._u_w.append(mx.sym.Variable("u_w_%d_weight" % layer_n))
            self._r_w.append(mx.sym.Variable("r_w_%d_weight" % layer_n))
            self._t_w.append(mx.sym.Variable("t_w_%d_weight" % layer_n))
            self._u_b.append(mx.sym.Variable("u_b_%d_bias" % layer_n))
            self._r_b.append(mx.sym.Variable("r_b_%d_bias" % layer_n))
            self._t_b.append(mx.sym.Variable("t_b_%d_bias" % layer_n))

        self._ctx = ctx
        self._domain_size = domain_size
        self._kernel_size = kernel_size
        self._num_maps = num_maps
        self._num_layers = num_layers
        self._input_transformer = input_transformer

        self._stored_weights = None


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

    def train(self, input_size, width, depth, X, y, batch_size, num_epochs, learning_rate, momentum, initializer, X_val=None, y_val=None, optimizer="adam", max_grad_norm=5.0, verbose=True, **kwargs):
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
        optimizer = mx.optimizer.create(optimizer, learning_rate=learning_rate, **kwargs)
        updater = mx.optimizer.get_updater(optimizer)

        inp = mx.sym.Variable('input')
        label = mx.sym.Variable('label')
        model_sym = self._build_neural_gpu(inp, label, batch_size, input_size, width, depth)

        arg_names = model_sym.list_arguments()

        # most of the code here is from the LSTM tutorial
        arg_shape, out_shape, aux_shape = model_sym.infer_shape(**self._shapes)
        args_grad = {}
        #print "ARG SHAPE", zip(arg_names, arg_shape)
        #print "OUT SHAPE", out_shape
        assert (initializer is None) == (self._stored_weights is not None)
        arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape] if initializer is not None else [self._stored_weights[n] if self.is_param_name(n) else mx.nd.zeros(s, ctx) for (n, s) in zip(arg_names, arg_shape)]

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
        for epoch_n in range(num_epochs):
            if verbose:
                print "Epoch %3d ========" % epoch_n,
            for it in range((num_samples + batch_size - 1) // batch_size):
                #print "Iteration %3d ----" % it
                mx.nd.array(NeuralGPU.get_batch(X, beginning, batch_size)).copyto(inp)
                mx.nd.array(NeuralGPU.get_batch(y, beginning, batch_size)).copyto(label)

                model_exec.forward(is_train=True)
                model_exec.backward()

                # updare parameters
                norm = 0.
                for idx, weight, grad, name in param_blocks:
                    grad /= batch_size
                    l2_norm = mx.nd.norm(grad).asscalar()
                    norm += l2_norm*l2_norm
                norm = math.sqrt(norm)
                for idx, weight, grad, name in param_blocks:
                    if norm > max_grad_norm:
                        grad *= (max_grad_norm / norm)
                    updater(idx, grad, weight)
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
                print a, b, 1.0 * a / b


        self._stored_weights = arg_dict


import random
if __name__ == "__main__":
    def to_bin(a, out, l, onehot=False):
        for i in range(l):
            bit = a % 2
            if onehot: out.append([bit, 1 - bit, 0])
            else: out.append(bit)
            a //= 2

    ctx = mx.gpu(0)
    ng = NeuralGPU(ctx, 3, TransformInputAcrossMaps)

    debugging_small_set = False

    num_epochs = 3 if not debugging_small_set else 1

    first = True
    for global_len in range(5, 21):
        for cur_len in [global_len, global_len, global_len, 10, global_len, global_len, global_len, 20, global_len, global_len, global_len + 1, global_len, 20]:
            print "CUR LEN ", cur_len
            X = []
            y = []
            X_val = []
            y_val = []
            total_samples = 5000 if not debugging_small_set else 400
            test_samples = 200
            for i in range(total_samples):
                a = random.randint(1, 1 << cur_len)
                b = random.randint(1, 1 << cur_len)
                c = a * b
                inp = []
                out = []
                to_bin(a, inp, cur_len)
                inp.append(2)
                to_bin(b, inp, cur_len)
                to_bin(c, out, cur_len + cur_len + 1)
                if i < total_samples - test_samples:
                    X.append(inp)
                    y.append(out)
                else:
                    X_val.append(inp)
                    y_val.append(out)

            ng.train(cur_len + cur_len + 1, 4, cur_len + cur_len, X, y, 128, num_epochs, 0.01, 0.9, mx.initializer.Uniform(0.3) if first else None, X_val=X_val, y_val=y_val)
            first = False

