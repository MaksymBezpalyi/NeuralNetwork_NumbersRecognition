import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probability = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probability

class Loss:
    def calculate(self,output, y):
        sample_losses = self.forward(output, y)
        data_losses = np.mean(sample_losses)
        return data_losses

class Loss_Categorical_Cross_Entropy(Loss): # loss prediction is the final output layer
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confiedences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confiedences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confiedences)

        return negative_log_likelihoods

    def backwards(self, y_true, y_pred):
        samples = len(y_pred)
        self.dinputs = y_pred.copy()

        for i in range(samples):
            true_index = y_true[i] # won't work with matrixes(just for MNIST)
            self.dinputs[i][true_index] -= 1

        self.dinputs = self.dinputs / len(y_pred)

class Optimizer:
    def __init__(self, learning_rate=0.1):#-----------------------------------------------
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

dense1 = Layer_Dense(784, 64)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(64,10)
activation2 = Activation_Softmax()

if __name__ == "__main__":
    from keras.datasets import mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    limit = 40000
    X = X_train[:limit].reshape(limit, -1).astype("float32") / 255
    y = y_train[:limit]

    loss_function = Loss_Categorical_Cross_Entropy()

    optimizer = Optimizer(learning_rate=0.5)

    for epoch in range(1001):

        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        predictions = np.argmax(activation2.output, axis=1)
        accuracy = np.mean(predictions == y)
        loss = loss_function.calculate(activation2.output, y)

        if epoch % 100 == 0:
            print(f'Epoch: {epoch}, loss: {loss:.3f}, acc: {accuracy:.3f}')

        loss_function.backwards(y, activation2.output)
        dense2.backward(loss_function.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.update_params(dense1)
        optimizer.update_params(dense2)

    np.save('w1.npy', dense1.weights)
    np.save('b1.npy', dense1.biases)
    np.save('w2.npy', dense2.weights)
    np.save('b2.npy', dense2.biases)