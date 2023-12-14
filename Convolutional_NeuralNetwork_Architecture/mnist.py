import tensorflow as tf

# Load and preprocess MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255, x_test / 255
y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

# Reshape and cast images to tensors
x_image_train = tf.cast(tf.reshape(x_train, [-1, 28, 28, 1]), "float32")
x_image_test = tf.cast(tf.reshape(x_test, [-1, 28, 28, 1]), "float32")

# Create datasets for training and testing
train_dataset = tf.data.Dataset.from_tensor_slices((x_image_train, y_train)).batch(50)
test_dataset = tf.data.Dataset.from_tensor_slices((x_image_test, y_test)).batch(50)

# Slice dataset for faster training (optional)
x_image_train = tf.slice(x_image_train, [0, 0, 0, 0], [10000, 28, 28, 1])
y_train = tf.slice(y_train, [0, 0], [10000, 10])

# Define the first convolutional layer
w_conv1 = tf.Variable(tf.random.truncated_normal([5, 5, 1, 32], stddev=0.1, seed=0))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

def filter_map(input_ft):
    """
    Apply the filter kernel to the input image.
    
    Args:
        input_ft: Input feature or image tensor.
    
    Returns:
        tf.nn.conv2d: Tensor representing the result of the convolution operation.
    """
    return tf.nn.conv2d(input_ft, w_conv1, padding="SAME", strides=[1, 1, 1, 1]) + b_conv1

def relu1(input_ft):
    """
    Apply ReLU activation function to the filter map.
    
    Args:
        input_ft: Input tensor.
    
    Returns:
        tf.nn.relu: Tensor with ReLU activation applied.
    """
    return tf.nn.relu(filter_map(input_ft))

def max_pool1(input_ft):
    """
    Apply max pooling to the ReLU-activated data.
    
    Args:
        input_ft: Input tensor.
    
    Returns:
        tf.nn.max_pool: Tensor with max pooling applied.
    """
    return tf.nn.max_pool(relu1(input_ft), ksize=[1, 2, 2, 1], padding="SAME")
