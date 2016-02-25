from jpype_connect import jpype_compile, register_java_class_path, find_jars
from deepstream_directory import locate_class_dir
from general.numpy_helpers import get_rng
import numpy as np
from utils.predictors.i_predictor import IPredictor
import jpype as jp

__author__ = 'peter'


# print locate_class_dir()

register_java_class_path(locate_class_dir())
# register_java_class_path('/Users/peter/projects/argmaxlab/argmaxlab/spiking_experiments')
# register_java_class_path(list(find_jars('/Users/peter/.m2/')))


class JavaSpikingNetWrapper(IPredictor):

    def __init__(self, ws, eta, n_steps, depth_first=False, fractional = False, queue_implementation = True,
                 return_counts = False, smooth_grads = False, forward_discretize = 'rect-herding',
                 back_discretize = 'herding', test_discretize = 'rect-herding', seed = 1234, regularization=0.,
                 hold_error = False, dtype = 'DOUBLE'):

        self.dtype = dtype;
        self.jnet = \
                jp.JClass('nl.uva.deepstream.mlp.SpikingMLP')(ws, n_steps, eta, depth_first, fractional, queue_implementation, return_counts,
                    smooth_grads, forward_discretize, back_discretize, test_discretize, seed, regularization, hold_error, dtype.upper(),
                    )

    def predict(self, x):
        prediction = np.array([a[:] for a in self.jnet.predict(x.astype(np.double))[:]])
        return prediction

    def predict_one(self, x):
        return np.array(self.jnet.predict_one(x.astype('float', copy=False)))[:]

    def train(self, x, targ):
        self.jnet.train(x.astype('float', copy=False), targ.astype('float', copy=False))

    @staticmethod
    def from_init(w_init, layer_sizes, rng=None, **init_args):
        rng = get_rng(rng)
        ws = [w_init*rng.randn(n_in, n_out) for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        return JavaSpikingNetWrapper(ws, **init_args)


if __name__ == '__main__':
    # Just to check if the imports work and you can run
    from jpype_connect import JPypeConnection
    with JPypeConnection():
        JavaSpikingNetWrapper.from_init(0.01, eta = 0.01, n_steps=10, layer_sizes = [10, 10, 10])
    print 'It works.'
