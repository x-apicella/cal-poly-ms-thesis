# models.py

from keras import layers, models
from keras.regularizers import l2
from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization, MultiHeadAttention, LayerNormalization, Dense
from keras.layers import MultiHeadAttention, LayerNormalization, Dense, Dropout, GlobalAveragePooling1D
import tensorflow as tf

def build_cnn(input_shape, num_classes, l2_reg=0.001, activation='softmax', **kwargs):
    """
    Builds a Convolutional Neural Network (CNN) with two Conv1D layers per block and MaxPooling.
    """
    
    # Retrieve parameters from kwargs or use defaults
    filters = kwargs.get('filters', [32, 64, 128])
    kernel_sizes = kwargs.get('kernel_sizes', [3, 3, 3])
    dropout_rates = kwargs.get('dropout_rates', [0.1, 0.1, 0.1, 0.5])
    
    # Input Layer
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Iterate through each Conv1D block
    for i in range(len(filters)):
        # First Conv1D layer in the block
        x = Conv1D(filters=filters[i],
                   kernel_size=kernel_sizes[i],
                   activation='relu',
                   padding='same',
                   kernel_regularizer=l2(l2_reg))(x)
        
        # Second Conv1D layer in the block
        x = Conv1D(filters=filters[i],
                   kernel_size=kernel_sizes[i],
                   activation='relu',
                   padding='same',
                   kernel_regularizer=l2(l2_reg))(x)
        
        # Determine pool_size based on the number of filters
        if filters[i] == 32:
            pool_size = 3
        elif filters[i] == 64:
            pool_size = 2
        else:
            pool_size = 5
        
        # MaxPooling and Dropout
        x = MaxPooling1D(pool_size=pool_size)(x)
        x = Dropout(rate=dropout_rates[i])(x)
    
    # Flatten the output from convolutional layers
    x = Flatten()(x)
    
    # Dense Layer with Dropout
    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(rate=dropout_rates[-1])(x)
    
    # Output Layer
    outputs = Dense(num_classes, activation=activation)(x)
    
    # Create the model
    model = models.Model(inputs, outputs)
    
    return model

def conv_block(x, filters, kernel_size=3, stride=1, l2_reg=0.001):
    """
    Basic residual block for ResNet18 and ResNet34.

    Parameters:
    - x (tensor): Input tensor.
    - filters (int): Number of filters.
    - kernel_size (int): Kernel size for Conv layers.
    - stride (int): Stride for the first Conv layer.
    - l2_reg (float): L2 regularization factor.

    Returns:
    - tensor: Output tensor after applying the residual block.
    """
    shortcut = x

    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same',
                                 kernel_regularizer=l2(l2_reg))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

def bottleneck_block(x, filters, kernel_size=3, stride=1, l2_reg=0.001):
    """
    Bottleneck residual block for ResNet50.

    Parameters:
    - x (tensor): Input tensor.
    - filters (int): Number of filters for the first two Conv layers.
    - kernel_size (int): Kernel size for the middle Conv layer.
    - stride (int): Stride for the first Conv layer.
    - l2_reg (float): L2 regularization factor.

    Returns:
    - tensor: Output tensor after applying the bottleneck block.
    """
    shortcut = x

    # 1x1 Conv
    x = layers.Conv1D(filters, 1, strides=stride, padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 3x3 Conv
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 1x1 Conv
    x = layers.Conv1D(filters * 4, 1, strides=1, padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters * 4:
        shortcut = layers.Conv1D(filters * 4, 1, strides=stride, padding='same',
                                 kernel_regularizer=l2(l2_reg))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

def build_resnet18_1d(input_shape, num_classes, l2_reg=0.001, activation='softmax', **kwargs):
    """
    Builds ResNet18 architecture for 1D data.

    Parameters:
    - input_shape (tuple): Shape of the input data (timesteps, features).
    - num_classes (int): Number of output classes.
    - l2_reg (float): L2 regularization factor.
    - activation (str): Activation function for the output layer ('softmax' or 'sigmoid').
    - **kwargs: Additional keyword arguments (ignored for ResNet18).

    Returns:
    - model (keras.Model): Compiled ResNet18 model.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 7, strides=2, padding='same',
                      kernel_regularizer=l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    for _ in range(2):
        x = conv_block(x, filters=64, l2_reg=l2_reg)

    for _ in range(2):
        stride = 2 if _ == 0 else 1
        x = conv_block(x, filters=128, stride=stride, l2_reg=l2_reg)

    for _ in range(2):
        stride = 2 if _ == 0 else 1
        x = conv_block(x, filters=256, stride=stride, l2_reg=l2_reg)

    for _ in range(2):
        stride = 2 if _ == 0 else 1
        x = conv_block(x, filters=512, stride=stride, l2_reg=l2_reg)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation=activation)(x)
    model = models.Model(inputs, outputs)
    return model

def build_resnet34_1d(input_shape, num_classes, l2_reg=0.001, activation='softmax', **kwargs):
    """
    Builds ResNet34 architecture for 1D data.

    Parameters:
    - input_shape (tuple): Shape of the input data (timesteps, features).
    - num_classes (int): Number of output classes.
    - l2_reg (float): L2 regularization factor.
    - activation (str): Activation function for the output layer ('softmax' or 'sigmoid').
    - **kwargs: Additional keyword arguments (ignored for ResNet34).

    Returns:
    - model (keras.Model): Compiled ResNet34 model.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 7, strides=2, padding='same',
                      kernel_regularizer=l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    for _ in range(3):
        x = conv_block(x, filters=64, l2_reg=l2_reg)

    for _ in range(4):
        stride = 2 if _ == 0 else 1
        x = conv_block(x, filters=128, stride=stride, l2_reg=l2_reg)

    for _ in range(6):
        stride = 2 if _ == 0 else 1
        x = conv_block(x, filters=256, stride=stride, l2_reg=l2_reg)

    for _ in range(3):
        stride = 2 if _ == 0 else 1
        x = conv_block(x, filters=512, stride=stride, l2_reg=l2_reg)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation=activation)(x)
    model = models.Model(inputs, outputs)
    return model

def build_resnet50_1d(input_shape, num_classes, l2_reg=0.001, activation='softmax', **kwargs):
    """
    Builds ResNet50 architecture for 1D data.

    Parameters:
    - input_shape (tuple): Shape of the input data (timesteps, features).
    - num_classes (int): Number of output classes.
    - l2_reg (float): L2 regularization factor.
    - activation (str): Activation function for the output layer ('softmax' or 'sigmoid').
    - **kwargs: Additional keyword arguments (ignored for ResNet50).

    Returns:
    - model (keras.Model): Compiled ResNet50 model.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 7, strides=2, padding='same',
                      kernel_regularizer=l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    for _ in range(3):
        x = bottleneck_block(x, filters=64, l2_reg=l2_reg)

    for _ in range(4):
        stride = 2 if _ == 0 else 1
        x = bottleneck_block(x, filters=128, stride=stride, l2_reg=l2_reg)

    for _ in range(6):
        stride = 2 if _ == 0 else 1
        x = bottleneck_block(x, filters=256, stride=stride, l2_reg=l2_reg)

    for _ in range(3):
        stride = 2 if _ == 0 else 1
        x = bottleneck_block(x, filters=512, stride=stride, l2_reg=l2_reg)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation=activation)(x)
    model = models.Model(inputs, outputs)
    return model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Remove explicit casting
    x = inputs
    
    # Multi-head self attention
    x = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=head_size, dropout=dropout
    )(x, x)
    
    # Add & normalize
    x = tf.keras.layers.Add()([x, inputs])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-forward network
    x_dense = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation="relu"),
        tf.keras.layers.Dense(inputs.shape[-1]),
    ])(x)
    
    # Add & normalize
    x = tf.keras.layers.Add()([x_dense, x])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    return x

def build_transformer(input_shape, num_classes, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0, activation='softmax', **kwargs):
    # The **kwargs allows the function to accept additional arguments without error
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(num_classes, activation=activation)(x)
    return models.Model(inputs, outputs)

def get_model_summary_table(model):
    """
    Returns a formatted string containing model architecture details in table format.
    Can be used by any model type (CNN, ResNet, Transformer).
    
    Args:
        model: Keras model object
    Returns:
        str: Formatted table string
    """
    # Initialize table string
    table = "\nModel Architecture Table:\n"
    table += "-" * 85 + "\n"
    table += f"{'Layer #':<8} {'Layer (type)':<25} {'Output Shape':<20} {'Param #':<12}\n"
    table += "-" * 85 + "\n"
    
    # Add each layer's information
    for i, layer in enumerate(model.layers):
        name = f"{layer.name} ({layer.__class__.__name__})"
        shape = str(layer.output_shape)
        params = layer.count_params()
        table += f"{i+1:<8} {name:<25} {shape:<20} {params:<12}\n"
    
    # Add total parameters
    table += "-" * 85 + "\n"
    trainable_params = sum([layer.count_params() for layer in model.layers if len(layer.weights) > 0])
    table += f"Total trainable parameters: {trainable_params:,}\n"
    table += "-" * 85 + "\n"
    
    return table