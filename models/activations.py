from tf_imports import activations

elu = activations.elu
selu = activations.selu
softsign = activations.softsign
softplus = activations.softplus
softmax = activations.softmax
tanh = activations.tanh
sigmoid = activations.sigmoid
hard_sigmoid = activations.hard_sigmoid
linear = activations.linear


def relu(x):
    return activations.relu(x, alpha=0, max_value=1)


def lrelu(x):
    return activations.relu(x, alpha=0.1)
