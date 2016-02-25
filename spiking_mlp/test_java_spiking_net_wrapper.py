from jpype_connect import JPypeConnection
from java_spiking_net_wrapper import JavaSpikingNetWrapper
from general.ezprofile import EZProfiler
from utils.benchmarks.predictor_comparison import assess_online_predictor
from utils.datasets.mnist import get_mnist_dataset
from utils.predictors.predictor_tests import assert_online_predictor_not_broken

__author__ = 'peter'


def test_java_spiking_net_wrapper():

    with JPypeConnection():
        assert_online_predictor_not_broken(
            predictor_constructor = lambda n_dim_in, n_dim_out:
                JavaSpikingNetWrapper.from_init(
                    layer_sizes = [n_dim_in, 100, n_dim_out],
                    n_steps = 10,
                    w_init = 0.01,
                    rng = 1234,
                    eta = 0.01,
                    ),
            initial_score_under=50,
            categorical_target=False,
            minibatch_size=1,
            n_epochs=2
            )


def profile_java_net():
    """

    Note: These times are super unreliable for some reason.. A given run can vary
    by 7s-14s for example.  God knows why.

    Version 'old', Best:
    Scores at Epoch 0.0: Test: 8.200
    Scores at Epoch 1.0: Test: 57.100
    Scores at Epoch 2.0: Test: 71.200
    Elapsed time is: 7.866s

    Version 'arr', Best:
    Scores at Epoch 0.0: Test: 8.200
    Scores at Epoch 1.0: Test: 58.200
    Scores at Epoch 2.0: Test: 71.500
    Elapsed time is: 261.1s

    Version 'new', Best:
    Scores at Epoch 0.0: Test: 8.200
    Scores at Epoch 1.0: Test: 58.200
    Scores at Epoch 2.0: Test: 71.500
    Elapsed time is: 8.825s

    :return:
    """

    mnist = get_mnist_dataset(flat=True).shorten(1000).to_onehot()

    with JPypeConnection():

        spiking_net = JavaSpikingNetWrapper.from_init(
            fractional = True,
            depth_first=False,
            smooth_grads = False,
            back_discretize = 'noreset-herding',
            w_init=0.01,
            hold_error=True,
            rng = 1234,
            n_steps = 10,
            eta=0.01,
            layer_sizes=[784]+[200]+[10],
            dtype = 'float'
            )

        with EZProfiler(print_result=True):
            result = assess_online_predictor(
                predictor = spiking_net,
                dataset=mnist,
                evaluation_function='percent_argmax_correct',
                test_epochs=[0, 1, 2],
                minibatch_size=1,
                test_on='test',
                )


if __name__ == '__main__':

    # profile_java_net()
    test_java_spiking_net_wrapper()
