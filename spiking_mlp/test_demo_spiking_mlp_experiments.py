from general.test_mode import set_test_mode
from demo_spiking_mlp_experiments import ExperimentLibrary

__author__ = 'peter'


def test_mnist_relu_vs_spiking():
    set_test_mode(True)
    ExperimentLibrary.mnist_relu_vs_spiking.run()


def test_try_hyperparams():
    set_test_mode(True)
    ExperimentLibrary.try_hyperparams.run()


if __name__ == '__main__':
    test_mnist_relu_vs_spiking()
    test_try_hyperparams()
