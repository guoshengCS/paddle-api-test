import collections
import inspect
import six
import sys

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers.utils as utils
from paddle.fluid.layers.utils import map_structure, flatten, pack_sequence_as
from paddle.fluid.dygraph import to_variable


class Model(fluid.dygraph.Layer):
    """
    graph: network + loss + optimizer + metric
    """
    def __init__(self, **kwargs):
        super(Model, self).__init__("model")
        self._dygraph_mode = False

        # extract data desc from method arguments
        self._data_descs = {}
        is_sequence_ori = utils.is_sequence
        # nested structure of shapes
        utils.is_sequence = self._InputDesc._is_shape_sequence
        for func in [self.forward, self.loss]:
            func_argspec = inspect.getargspec(func)
            for i, arg in enumerate(func_argspec.args[::-1]):
                if arg.endswith("_shape"):
                    assert i <= len(
                        func_argspec.defaults
                    ), "The shape argument must have default value."
                    self._data_descs[arg[:-len("_shape")]] = map_structure(
                        lambda shape: self._InputDesc(shape),
                        func_argspec.defaults[-i - 1])
        utils.is_sequence = is_sequence_ori
        print(self._data_descs)


        # self._optimizer = kwargs.get()
        # TODO: mutable program
        # self._cache_programs = {}

    class _InputDesc(object):
        def __init__(self, shape):
            self.shape = shape

        @staticmethod
        def _is_shape_sequence(seq):
            if sys.version_info < (3, ):
                integer_types = (int, long)
            else:
                integer_types = (int, )
            if (isinstance(seq, list) or isinstance(seq, tuple)):
                if reduce(
                        lambda flag, x:
                    (x is None or isinstance(x, integer_types)) and flag, seq,
                        True):
                    return False
            if isinstance(seq, dict):
                return True
            return (isinstance(seq, collections.Sequence)
                    and not isinstance(seq, six.string_types))

        def __str__(self):
            return self.__repr__()

        def __repr__(self):
            return "shape: {}".format(self.shape)

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

        model_self = self

        class _DygraphCaller(object):
            def __init__(self, functions, for_test, *args, **kwargs):
                self._functions = functions
                self._for_test = for_test

            def _convert_input(self, input):
                return map_structure(
                    lambda x: to_variable(x)
                    if isinstance(x, np.ndarray) else x, input)

            def _convert_args(self, function, args, kwargs):
                function_argspec = inspect.getargspec(function)
                function_argspec_args = function_argspec.args
                if inspect.ismethod(function):
                    # exclude implicit 'self' or 'cls' argument
                    function_argspec_args = function_argspec_args[1:]
                function_kwargs = {}
                end_idx = len(function_argspec_args)
                # convert named arguments
                for arg_idx, arg in enumerate(function_argspec_args):
                    if kwargs.has_key(arg):
                        if end_idx == len(function_argspec_args):
                            # truncate args from the first keyword argument
                            end_idx = arg_idx
                        function_kwargs[arg] = self._convert_input(
                            kwargs.pop(arg))
                # convert defaults
                # It is ambiguous to split args if args has default values.
                # Use defaults only for args after the first named argument,
                # while args before the first named argument should also be
                # able to use defaults.
                if function_argspec.defaults:
                    for arg_idx, arg in enumerate(
                            function_argspec_args[end_idx:][::-1]):
                        if not function_kwargs.has_key(arg):
                            function_kwargs[arg] = self._convert_input(
                                function_argspec.defaults[-arg_idx - 1])
                    end_idx -= len(function_argspec.defaults)
                # convert positional arguments
                function_args = [self._convert_input(x) for x in args[:end_idx]]
                return function_args, function_kwargs, args[end_idx:], kwargs

            def __call__(self, *args, **kwargs):
                for function in self._functions:
                    (function_args, function_kwargs, remain_args,
                     remain_kwargs) = self._convert_args(
                         function, args, kwargs)
                    function_outs = function(*function_args, **function_kwargs)
                    args = list(function_outs if isinstance(
                        function_outs, collections.Sequence
                    ) else [function_outs]) + list(remain_args)
                    kwargs = remain_kwargs
                return function_outs

        class _GraphCaller(object):
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

            def _convert_input(self,
                               input,
                               input_name,
                               input_idx,
                               is_default=False):
                def _to_variable(x, x_desc=None, x_name=None, x_idx=None):
                    if isinstance(x, np.ndarray):
                        out = fluid.data(name=x_name if x_idx is None else
                                         (x_name + "_" + str(x_idx)),
                                         shape=([None] + list(x.shape[1:]))
                                         if x_desc is None else x_desc.shape,
                                         dtype=x.dtype)
                        # set the way to get input data, then we can use it to
                        # extract data from args and kwargs when running __call__
                        if is_default:  # for defaults
                            if x_idx is None:  # if input is plain
                                data_extracter = lambda args, kwargs: input
                            else:  #  if input is nested structure
                                data_extracter = lambda args, kwargs: flatten(
                                    input)[x_idx]
                        elif input_idx is None:  # for named arg
                            if x_idx is None:  # if input is plain
                                data_extracter = lambda args, kwargs: kwargs[
                                    input_name]
                            else:  #  if input is nested structure
                                data_extracter = lambda args, kwargs: flatten(
                                    kwargs[input_name])[x_idx]
                        else:  # for positional arg
                            if x_idx is None:  # if input is plain
                                data_extracter = lambda args, kwargs: args[
                                    input_idx]
                            else:  #  if input is nested structure
                                data_extracter = lambda args, kwargs: flatten(
                                    args[input_idx])[x_idx]
                        self._inputs[out.name] = data_extracter
                    else:
                        out = x
                    return out

                input_desc = model_self._data_descs.get(input_name, None)
                if not utils.is_sequence(input):
                    return _to_variable(input, input_desc, input_name)
                flat_output = []
                if input_desc is None:
                    for i, x in enumerate(flatten(input)):
                        out = _to_variable(x, x_name=input_name, x_idx=i)
                        flat_output.append(out)
                else:
                    for i, x in enumerate(
                            zip(flatten(input), flatten(input_desc))):
                        out = _to_variable(*x, x_name=input_name, x_idx=i)
                        flat_output.append(out)
                output = pack_sequence_as(input, flat_output)
                return output

            def _convert_args(self, function, args, kwargs):
                # print(inspect.getcallargs(function, *args, **kwargs))
                function_argspec = inspect.getargspec(function)
                function_argspec_args = function_argspec.args
                if inspect.ismethod(function):
                    # exclude implicit 'self' or 'cls' argument
                    function_argspec_args = function_argspec_args[1:]
                function_kwargs = {}
                end_idx = len(function_argspec_args)
                # convert named arguments
                for arg_idx, arg in enumerate(function_argspec_args):
                    if kwargs.has_key(arg):
                        if end_idx == len(function_argspec_args):
                            # truncate args from the first keyword argument
                            end_idx = arg_idx
                        function_kwargs[arg] = self._convert_input(
                            kwargs.pop(arg), input_name=arg, input_idx=None)
                # convert defaults
                # It is ambiguous to split args if args has default values.
                # Use defaults only for args after the first named argument,
                # while args before the first named argument should also be
                # able to use defaults.
                if function_argspec.defaults:
                    for arg_idx, arg in enumerate(
                            function_argspec_args[end_idx:][::-1]):
                        if not function_kwargs.has_key(arg):
                            function_kwargs[arg] = self._convert_input(
                                function_argspec.defaults[-arg_idx - 1],
                                input_name=arg,
                                input_idx=None,
                                is_default=True)
                    end_idx -= len(function_argspec.defaults)
                # convert positional arguments
                function_args = [  # actually we needn't use arg name as var name
                    self._convert_input(i, *arg) for i, arg in enumerate(
                        zip(function_argspec_args[:end_idx], args[:end_idx]))
                ]
                return function_args, function_kwargs, args[end_idx:], kwargs

            def __call__(self, *args, **kwargs):
                # since we only run functions one time to build graph,
                # assume all run has same arg signature with the first run
                # TODO: check all run has the same arg signature
                outputs = self._executor.run(
                    self._main_program,
                    feed=dict([
                        (var_name, data_extracter(args, kwargs))
                        for var_name, data_extracter in self._inputs.items()
                    ]),
                    fetch_list=self._outputs)
                return outputs

        if self._dygraph_mode:
            return _DygraphCaller(functions, for_test, *args, **kwargs)
        else:
            return _GraphCaller(functions, for_test, *args, **kwargs)

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

    def forward(self, x, y, x_shape=[None, 8]):
        # print("x: ", x)
        # print("y: ", y)
        print("x_shape: ", x_shape)
        x = x + y
        # print("x+y: ", x)
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
    # print(my_model(
    #     np.random.rand(2, 8).astype("float32"),
    #     np.random.rand(2, 8).astype("float32")))
    print(my_model.train(np.random.rand(2, 8).astype("float32"),
                         np.random.rand(2, 8).astype("float32"),
                         target=np.random.rand(2, 1).astype("float32")))
    # fc = fluid.dygraph.FC("test", 1)
    # pred = fc(to_variable(np.random.rand(2, 8).astype("float32")))
    # loss = fluid.layers.square_error_cost(
    #     pred,
    #     to_variable(np.random.rand(2, 1).astype("float32")))
    # optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    # x = optimizer.minimize(loss)
    # print(x)
print(flatten([np.random.rand(2, 3), np.random.rand(2, 3)]))
print(map_structure(lambda x: x, np.random.rand(2, 3)))
print(flatten(Model._InputDesc([None, 1,2,3])))
print(len(flatten(Model._InputDesc([None, 1, 2, 3]))))

def tmp(a, b=1):pass
# print(inspect.getargspec(tmp))
print(map_structure(tmp, [1]))
print(map_structure(tmp, [1], [1]))
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