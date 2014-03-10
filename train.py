import pylab as pl
import numpy as np

from positive_class_classifier import PositiveClassClassifier


######################################################################
# Plot results
######################################################################

def plot_accuracy(x, y, plot_placement, x_legend):
    """Plot accuracy as a function of x."""
    x = np.array(x)
    y = np.array(y)
    pl.subplots_adjust(hspace=0.5)
    pl.subplot(plot_placement)
    pl.title('Classification accuracy as a function of %s' % x_legend)
    pl.xlabel('%s' % x_legend)
    pl.ylabel('Accuracy')
    pl.grid(True)
    pl.plot(x, y)

def train_PCC():
    classifier = PositiveClassClassifier('javascript')
    classifier.train()

    pl.figure(1)

    # Plot accuracy evolution with #examples
    accuracy, n_examples = zip(*classifier.stats['accuracy_history'])
    plot_accuracy(n_examples, accuracy, 211, "training examples (#)")

    # Plot accuracy evolution with runtime
    accuracy, runtime = zip(*classifier.stats['runtime_history'])
    plot_accuracy(runtime, accuracy, 212, 'runtime (s)')

    pl.show()

if __name__ == "__main__": 
    train_PCC()
