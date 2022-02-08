# Single layer perceptron to process images of alphabet letters

A collection of images were compiled for training and testing the perceptron. The file assign1_data1.h5 contains variables trainims (training images) and testims (testing images) along with the ground truth labels in trainlbls and testlbls.

## Part A

Visualize a sample image for each class. Find correlation coefficients between pairs of sample images selected. Display the correlations in matrix format. Discuss the degree of within-class versus across-class variability.

## Part B

Design a single-layer perceptron with an output neuron for each digit, using the training data. Set the initial network weights w and bias term b as random numbers drawn from a Gaussian distribution N(0,0.01), assume a sigmoid activation function. The implementation should not train each output neuron separately, but a compound matrix W and a compound vecor b should be defined and used to simultaneously update all connections. The online training algorithm should perform 10000 iterations. At each iteration, a sample image should be randomly selected from the training data, the network should be updated according to the gradient-descent learning rule, and W, b, and the mean-squared error (MSE) should be recorded. Tune the learning rate η in order to minimize the final value of the MSE. Display the final network weights for each digit as a separate image, and describe the visual characteristics.

## Part C

Now separately repeat the training process using a substantially higher and a subtantially lower value than η. On a single figure, plot the MSE curves (across all 10000 iterations) for η_high, η_low and η.

## Part D

Validate the performance of the trained networks using all samples in the test data. Report the performance values for the three networks with η_high, η_low and η.
