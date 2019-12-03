import inspect
import collections

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.layers.utils import map_structure, flatten
from paddle.fluid.dygraph import to_variable


class Model(fluid.dygraph.Layer):
    """
    graph: network + loss + optimizer + metric
    """
    def __init__(self, **kwargs):
        super(Model, self).__init__("model")
        self._dygraph_mode = True

        # extract data desc from method arguments
        self._data_descs = {}
        for func in [self.forward, self.loss]:
            func_argspec = inspect.getargspec(func)
            for i, arg in enumerate(func_argspec.args[::-1]):
                if arg.endswith("_shape"):
                    assert i <= len(
                        func_argspec.defaults
                    ), "The shape argument must have default value."
                    self._data_descs[arg[:-len("_shape")]] = self.InputDesc(
                        func_argspec.defaults[-i - 1])

    class InputDesc(collections.namedtuple("InputDesc", ("shape", ))):
        pass

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        raise NotImplementedError

    def optim(self, loss):
        raise NotImplementedError

    def _make_sequential_function(self, functions, for_test, *args, **kwargs):
        # TODO: add customed output to fetch

        model_class_self = self

        class _DygraphCallFunctor(object):
            def __init__(self, functions, for_test, *args,
                         **kwargs):
                self._functions = functions
                self._for_test = for_test

            def _convert_input(self, input):
                return map_structure(
                    lambda x: to_variable(x)
                    if isinstance(x, np.ndarray) else x, input)

            def _convert_args(self, function, args, kwargs):
                function_argspec = inspect.getargspec(function)
                function_kwargs = {}
                end_idx = len(function_argspec.args)
                # truncate args from the first keyword argument, otherwise
                # TypeError: got multiple values for keyword argument
                for arg_idx, arg in enumerate(function_argspec.args):
                    if kwargs.has_key(arg):
                        end_idx = arg_idx
                        function_kwargs[arg] = self._convert_input(
                            kwargs.pop(arg))
                function_args = [self._convert_input(x) for x in args[:end_idx]]
                return function_args, function_kwargs, args[end_idx:], kwargs

            def __call__(self, *args, **kwargs):
                for function in self._functions:
                    # print(function)
                    # print(args)
                    # print(kwargs)
                    (function_args, function_kwargs, remain_args,
                     remain_kwargs) = self._convert_args(
                         function, args, kwargs)
                    # print(function_args)
                    # print(function_kwargs)
                    function_outs = function(*function_args, **function_kwargs)
                    args = list(function_outs if isinstance(
                        function_outs, collections.Sequence
                    ) else [function_outs]) + list(remain_args)
                    kwargs = remain_kwargs
                return function_outs

        class _GraphCallFunctor(object):
            def __init__(self, functions, for_test, *args, **kwargs):
                self._functions = functions
                self._for_test = for_test
                self._executor = fluid.Executor(fluid.CPUPlace())
                self._main_program = fluid.Program()
                self._startup_program = fluid.Program()
                self._inputs = []
                with fluid.program_guard(self._main_program,
                                         self._startup_program):
                    with fluid.unique_name.guard():
                        for function in self._functions:
                            # print(function)
                            # print(args)
                            # print(kwargs)
                            (function_args, function_kwargs, remain_args,
                             remain_kwargs) = self._convert_args(
                                 function, args, kwargs)
                            # print(function_args)
                            # print(function_kwargs)
                            function_outs = function(*function_args,
                                                     **function_kwargs)
                            args = list(function_outs if isinstance(
                                function_outs, collections.Sequence
                            ) else [function_outs]) + list(remain_args)
                            kwargs = remain_kwargs
                self._outputs = function_outs
                if for_test:
                    self._main_program = self._main_program.clone(for_test=True)
                # initialization
                for var in [
                        var for var in self._startup_program.list_vars()
                        if hasattr(var, "initializer")
                ]:
                    if not fluid.global_scope().find_var(
                            var.name).get_tensor()._is_initialized():
                        self._executor.run(self._startup_program)
                        break

            def _convert_input(self, input):
                model_class_self._data_descs
                return map_structure(
                    lambda x: to_variable(x)
                    if isinstance(x, np.ndarray) else x, input)

            def _convert_args(self, function, args, kwargs):
                function_argspec = inspect.getargspec(function)
                function_kwargs = {}
                end_idx = len(function_argspec.args)
                # truncate args from the first keyword argument, otherwise
                # TypeError: got multiple values for keyword argument
                for arg_idx, arg in enumerate(function_argspec.args):
                    if kwargs.has_key(arg):
                        end_idx = arg_idx
                        function_kwargs[arg] = self._convert_input(
                            kwargs.pop(arg))
                function_args = [self._convert_input(x) for x in args[:end_idx]]
                return function_args, function_kwargs, args[end_idx:], kwargs

            def __call__(self, *args, **kwargs):
                outputs = self._executor.run(self._main_program,
                                             feed=self._inputs,
                                             fetch_list=self._outputs)
                return outputs

        if self._dygraph_mode:
            return _DygraphCallFunctor(functions, for_test, *args, **kwargs)
        else:
            return _GraphCallFunctor(functions, for_test, *args, **kwargs)

    def train(self, *args, **kwargs):
        if not hasattr(self, "_train_function"):
            self._train_function = self._make_sequential_function(
                [self.forward, self.loss, self.optim], False, *args, **kwargs)
        return self._train_function(*args, **kwargs)

    def test(self, *args, **kwargs):
        if not hasattr(self, "_test_function"):
            self._test_function = self._make_sequential_function(
                [self.forward, self.loss], True, *args, **kwargs)
        return self._test_function(*args, **kwargs)

    def predict(self, *args, **kwargs):
        if not hasattr(self, "_predict_function"):
            self._predict_function = self._make_sequential_function(
                [self.forward], True, *args, **kwargs)
        return self._predict_function(*args, **kwargs)

    # def train(self, x, y=None, x_shape=None, y_shape=None):
    #     fluid.framework._dygraph_tracer().train_mode()

    # def eval(self):
    #     fluid.framework._dygraph_tracer().eval_mode()



class MyModel(Model):
    def __init__(self, name, hidden):
        super(MyModel, self).__init__()
        self.fc = fluid.dygraph.FC(name, hidden)

    def forward(self, x, x_shape=[None, 8]):
        # print("x: ", x)
        x = self.fc(x)
        return x

    def loss(self, pred, target):
        x = fluid.layers.reduce_mean(
            fluid.layers.square_error_cost(pred, target))
        return x

    def optim(self, loss):
        optimizer = fluid.optimizer.SGD(learning_rate=0.001)
        x = optimizer.minimize(loss)
        return x



with fluid.dygraph.guard():
    my_model = MyModel('myfc', 1)
    print(my_model(np.random.rand(2, 8).astype("float32")))
    print(my_model.train(np.random.rand(2, 8).astype("float32"),
                         target=np.random.rand(2, 1).astype("float32")))
    # fc = fluid.dygraph.FC("test", 1)
    # pred = fc(to_variable(np.random.rand(2, 8).astype("float32")))
    # loss = fluid.layers.square_error_cost(
    #     pred,
    #     to_variable(np.random.rand(2, 1).astype("float32")))
    # optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    # x = optimizer.minimize(loss)
    # print(x)

exit(0)


def train():
    model = MyModel('myfc', 1)

    batch_size = 20
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.uci_housing.train(), buf_size=500),
        batch_size=batch_size)
    test_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.uci_housing.test(), buf_size=500),
        batch_size=batch_size)

    for epoch in range(100):
        for batch_id, data in enumerate(train_reader()):
            x_data = np.array([x[0] for x in data], dtype=np.float32)
            y_data = np.array([x[1] for x in data], dtype=np.float32)

            loss = model.train(x_data, y_data)

            if batch_id % 10 == 0 and batch_id is not 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(
                    epoch, batch_id, loss))

train()
