from datetime import datetime
from fileman.experiment_record import ExperimentLibrary, Experiment
from general.test_mode import is_test_mode, set_test_mode
from plato.tools.common.online_predictors import GradientBasedPredictor
from plato.tools.mlp.mlp import MultiLayerPerceptron
from plato.tools.optimization.optimizers import GradientDescent
from spiking_mlp.java_spiking_net_wrapper import JavaSpikingNetWrapper
from spiking_mlp.jpype_connect import with_jpype
from utils.benchmarks.plot_learning_curves import plot_learning_curves
from utils.benchmarks.predictor_comparison import compare_predictors, LearningCurveData
from utils.benchmarks.train_and_test import percent_argmax_incorrect
from utils.datasets.mnist import get_mnist_dataset
import matplotlib.pyplot as plt
import pickle
import numpy as np
import jpype as jp

__author__ = 'peter'


@with_jpype
def compare_spiking_to_nonspiking(hidden_sizes = [300, 300], eta=0.01, w_init=0.01, fractional = False, n_epochs = 20,
                                  forward_discretize = 'rect-herding', back_discretize = 'noreset-herding', test_discretize='rect-herding', save_results = False):

    mnist = get_mnist_dataset(flat=True).to_onehot()
    test_epochs=[0.0, 0.05, 0.1, 0.2, 0.5]+range(1, n_epochs+1)

    if is_test_mode():
        mnist = mnist.shorten(500)
        eta = 0.01
        w_init=0.01
        test_epochs = [0.0, 0.05, 0.1]

    spiking_net = JavaSpikingNetWrapper.from_init(
        fractional = fractional,
        depth_first=False,
        smooth_grads = False,
        forward_discretize = forward_discretize,
        back_discretize = back_discretize,
        test_discretize = test_discretize,
        w_init=w_init,
        hold_error=True,
        rng = 1234,
        n_steps = 10,
        eta=eta,
        layer_sizes=[784]+hidden_sizes+[10],
        )

    relu_net = GradientBasedPredictor(
        MultiLayerPerceptron.from_init(
            hidden_activation = 'relu',
            output_activation = 'relu',
            layer_sizes=[784]+hidden_sizes+[10],
            use_bias=False,
            w_init=w_init,
            rng=1234,
            ),
        cost_function = 'mse',
        optimizer=GradientDescent(eta)
        ).compile()

    # Listen for spikes
    forward_eavesdropper = jp.JClass('nl.uva.deepstream.eavesdroppers.SpikeCountingEavesdropper')()
    backward_eavesdropper = jp.JClass('nl.uva.deepstream.eavesdroppers.SpikeCountingEavesdropper')()
    for lay in spiking_net.jnet.layers:
        lay.forward_herder.add_eavesdropper(forward_eavesdropper)
    for lay in spiking_net.jnet.layers[1:]:
        lay.backward_herder.add_eavesdropper(backward_eavesdropper)
    spiking_net.jnet.error_counter.add_eavesdropper(backward_eavesdropper)
    forward_counts = []
    backward_counts = []

    def register_counts():
        forward_counts.append(forward_eavesdropper.get_count())
        backward_counts.append(backward_eavesdropper.get_count())

    results = compare_predictors(
        dataset=mnist,
        online_predictors={
            'Spiking-MLP': spiking_net,
            'ReLU-MLP': relu_net,
            },
        test_epochs=test_epochs,
        online_test_callbacks=lambda p: register_counts() if p is spiking_net else None,
        minibatch_size = 1,
        test_on = 'training+test',
        evaluation_function=percent_argmax_incorrect,
        )

    spiking_params = [np.array(lay.forward_weights.w.asFloat()).copy() for lay in spiking_net.jnet.layers]
    relu_params = [param.get_value().astype(np.float64) for param in relu_net.parameters]

    # See what the score is when we apply the final spiking weights to the
    offline_trained_spiking_net = JavaSpikingNetWrapper(
        ws=relu_params,
        fractional = fractional,
        depth_first=False,
        smooth_grads = False,
        forward_discretize = forward_discretize,
        back_discretize = back_discretize,
        test_discretize = test_discretize,
        hold_error=True,
        n_steps = 10,
        eta=eta,
        )

    # for spiking_layer, p in zip(spiking_net.jnet.layers, relu_params):
    #     spiking_layer.w = p.astype(np.float64)

    error = [
        ('Test', percent_argmax_incorrect(offline_trained_spiking_net.predict(mnist.test_set.input), mnist.test_set.target)),
        ('Training', percent_argmax_incorrect(offline_trained_spiking_net.predict(mnist.training_set.input), mnist.training_set.target))
        ]
    results['Spiking-MLP with ReLU weights'] = LearningCurveData()
    results['Spiking-MLP with ReLU weights'].add(None, error)
    print 'Spiking-MLP with ReLU weights: %s' % error
    # --------------------------------------------------------------------------

    # See what the score is when we plug the spiking weights into the ReLU net.
    for param, sval in zip(relu_net.parameters, spiking_params):
        param.set_value(sval)
    error = [
        ('Test', percent_argmax_incorrect(relu_net.predict(mnist.test_set.input), mnist.test_set.target)),
        ('Training', percent_argmax_incorrect(relu_net.predict(mnist.training_set.input), mnist.training_set.target))
        ]
    results['ReLU-MLP with Spiking weights'] = LearningCurveData()
    results['ReLU-MLP with Spiking weights'].add(None, error)
    print 'ReLU-MLP with Spiking weights: %s' % error
    # --------------------------------------------------------------------------

    if save_results:
        with open("mnist_relu_vs_spiking_results-%s.pkl" % datetime.now(), 'w') as f:
            pickle.dump(results, f)

    # Problem: this currently includes test
    forward_rates = np.diff(forward_counts) / (np.diff(test_epochs)*60000)
    backward_rates = np.diff(backward_counts) / (np.diff(test_epochs)*60000)

    plt.figure('ReLU vs Spikes')
    plt.subplot(211)
    plot_learning_curves(results, title = "MNIST Learning Curves", hang=False, figure_name='ReLU vs Spikes', xscale='linear', y_title='Percent Error')
    plt.subplot(212)
    plt.plot(test_epochs[1:], forward_rates)
    plt.plot(test_epochs[1:], backward_rates)
    plt.xlabel('Epoch')
    plt.ylabel('n_spikes')
    plt.legend(['Mean Forward Spikes', 'Mean Backward Spikes'], loc='best')
    plt.ioff()
    plt.show()
    # plt.ioff()


ExperimentLibrary.mnist_relu_vs_spiking = Experiment(
    description = "Compare a RELU network to a spiking network and plot the learning curves.",
    function = compare_spiking_to_nonspiking,
    versions = {
        'small': dict(hidden_sizes = [100], n_epochs = 4),
        'big': dict(hidden_sizes = [200, 200], n_epochs = 20),
        'big-frac': dict(hidden_sizes = [200, 200], n_epochs = 20, fractional = True),
        'big-slow': dict(hidden_sizes = [200, 200], n_epochs = 20, eta = 0.002),
        'big-noreset-training': dict(hidden_sizes = [200, 200], n_epochs = 20, forward_discretize='noreset-rect-herding', back_discretize = 'noreset-herding', test_discretize='rect-herding'),
        'big-rand-backward': dict(hidden_sizes = [200, 200], n_epochs = 20, forward_discretize='rect-herding', back_discretize = 'rand-shift-herding', test_discretize='rect-herding'),
    },
    current_version='big-frac',
    conclusion = """
        All results are shown as Test/Training

                                ReLU            Spiking         ReLU-SpikeWeights   Spiking ReLUWeigths
        big:                    1.81/0.23       2.68/1.588      2.79/1.678          1.92/0.254
        big-frac                  " / "         2.34/0.926      2.20/0.878          1.92/0.254
        big-noreset-training:     " / "         2.44/1.462      2.47/1.510          3.15/1.888
        big-rand-backward         " / "         2.76/1.684      2.88/1.742
    """
)


ExperimentLibrary.try_hyperparams = Experiment(
    description="Compare the various hyperparameters to the baseline.",
    function=with_jpype(lambda
            fractional = False,
            depth_first = False,
            smooth_grads = False,
            back_discretize = 'noreset-herding',
            n_steps = 10,
            hidden_sizes = [200, 200],
            hold_error = True,
            :
        compare_predictors(
            dataset=(get_mnist_dataset(flat=True).shorten(100) if is_test_mode() else get_mnist_dataset(flat=True)).to_onehot(),
            online_predictors={'Spiking MLP': JavaSpikingNetWrapper.from_init(
                fractional = fractional,
                depth_first = depth_first,
                smooth_grads = smooth_grads,
                back_discretize = back_discretize,
                w_init=0.01,
                rng = 1234,
                eta=0.01,
                n_steps = n_steps,
                hold_error=hold_error,
                layer_sizes=[784]+hidden_sizes+[10],
                )},
            test_epochs=[0.0, 0.05] if is_test_mode() else [0.0, 0.05, 0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
            minibatch_size = 1,
            report_test_scores=True,
            test_on = 'test',
            evaluation_function='percent_argmax_incorrect'
            )),
    versions={
        'Baseline': dict(),
        'Fractional-Updates': dict(fractional = True),
        'Depth-First': dict(depth_first = True),
        'Smooth-Grads': dict(smooth_grads = True),
        'BackQuant-Zero-Reset': dict(back_discretize='herding'),
        'BackQuant-Rand-Reset': dict(back_discretize='rand-shift-herding'),
        'T=5': dict(n_steps = 5),
        'T=20': dict(n_steps = 20),
        'Depth-First-nohold': dict(depth_first = True, hold_error = False),
        'Smooth&Fractional': dict(smooth_grads=True, fractional = True)
    },
    current_version="Depth-First-nohold",
    conclusion="""
        Baseline                3.38
        Fractional-Updates      3.10
        Smooth-Grads:           2.85
        Smooth+Fractional:      3.07 ... huh, I thought this didnt work
        Depth-First:            3.38 ... With hold-error, it behaves the same as breadth-first
        Depth-First-nohold:    81.47 ... Crash
        BackQuant-Zero-Reset:  87.87 ... Not surprizing - it's the no-learning problem
        BackQuant-Rand-Reset    3.15
        T=5                     4.41
        T=20                    2.65
    """
    )


if __name__ == '__main__':

    set_test_mode(True)
    experiment = ExperimentLibrary.try_hyperparams
    experiment.run()
