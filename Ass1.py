class McPN:
    def __init__(self, weights, threshold, inputs):
        # Ensure that the number of inputs and weights are the same
        if len(inputs) != len(weights):
            raise ValueError("Number of inputs should be equal to the number of weights")

        self.weights = weights
        self.threshold = threshold
        self.inputs = inputs

    def activation(self):
        # Calculate the weighted sum and compare it with the threshold
        weighted_sum = sum(w * x for w, x in zip(self.weights, self.inputs))
        return int(weighted_sum >= self.threshold)

# Initialize the neuron with weights, threshold, and input values
neuron_weights = [1, 1]
neuron_threshold = 2
input_values = [0, 1]

# Create an instance of McPN
mcPNeuron = McPN(neuron_weights, neuron_threshold, input_values)

# Get the output of the neuron
output = mcPNeuron.activation()

print("Output:", output)  
