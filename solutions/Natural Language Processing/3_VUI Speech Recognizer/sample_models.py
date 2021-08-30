from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def cnn_output_length_2(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    value = 0
    for filtersize in filter_size:
        dilated_filter_size = filtersize + (filtersize - 1) * (dilation - 1)
        if border_mode == 'same':
            output_length = input_length
        elif border_mode == 'valid':
            output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    for i in range(recur_layers):
        if i == 0:
            curr_name = 'rnn' + str(i + 1)
            rnn = GRU(units, activation='relu', return_sequences=True, implementation=2, name=curr_name)(input_data)
            bn_rnn = BatchNormalization(name='bn_' + curr_name)(rnn)
        else:
            curr_name = 'rnn' + str(i + 1)
            rnn = GRU(units, activation='relu', return_sequences=True, implementation=2, name=curr_name)(bn_rnn)
            bn_rnn = BatchNormalization(name='bn_' + curr_name)(rnn)
            
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu', return_sequences=True, implementation=2, name='bidir_rnn'))(input_data)
    bn_bidir = BatchNormalization(name='bn_bidir')(bidir_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(bn_bidir)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def model_no_5(input_dim, filters, conv_stride, conv_border_mode, units, output_dim=29):
    kernels = [11]
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    conv_1d = Conv1D(filters, kernels[0], strides=conv_stride, padding=conv_border_mode, activation='relu', name='conv_1d')(input_data)
    batch_norm_cnn_1 = BatchNormalization(name='batch_norm_conv_1d')(conv_1d)    

    simple_rnn_1 = SimpleRNN(units, activation='relu', return_sequences=True, implementation=2, name='simple_rnn_1')(batch_norm_cnn_1)
    batch_norm_rnn_1 = BatchNormalization(name='batch_norm_rnn_1')(simple_rnn_1)
    
    simple_rnn_2 = SimpleRNN(units, activation='relu', return_sequences=True, implementation=2, name='simple_rnn_2')(batch_norm_rnn_1)
    batch_norm_rnn_2 = BatchNormalization(name='batch_norm_rnn_2')(simple_rnn_2)
    
    time_distributed_dense_1 = TimeDistributed(Dense(100), name='time_distributed_dense_1')(batch_norm_rnn_2)
    batch_norm_time_distributed_dense_1 = BatchNormalization(name='batch_norm_time_distributed_dense_1')(time_distributed_dense_1)
    dropout_time_distributed_dense_1 = Dropout(.5, name='dropout_time_distributed_dense_1')(batch_norm_time_distributed_dense_1)

    time_distributed_dense_2 = TimeDistributed(Dense(50), name='time_distributed_dense_2')(dropout_time_distributed_dense_1)
    batch_norm_time_distributed_dense_2 = BatchNormalization(name='batch_norm_time_distributed_dense_2')(time_distributed_dense_2)
    dropout_time_distributed_dense_2 = Dropout(.5, name='dropout_time_distributed_dense_2')(batch_norm_time_distributed_dense_2)
    
    time_distributed_dense = TimeDistributed(Dense(output_dim), name='time_distributed_dense')(dropout_time_distributed_dense_2)
    y_pred = Activation('softmax', name='softmax')(time_distributed_dense)
    
    model = Model(inputs=input_data, outputs=y_pred)    
    model.output_length = lambda x: cnn_output_length_2(x, kernels, conv_border_mode, conv_stride)
    
    print(model.summary())
    
    return model


def final_model(input_dim, filters, conv_stride, conv_border_mode, units, output_dim=29):
    kernels = [11]
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    conv_1d = Conv1D(filters, kernels[0], strides=conv_stride, padding=conv_border_mode, activation='relu', name='conv_1d')(input_data)
    batch_norm_cnn_1 = BatchNormalization(name='batch_norm_conv_1d')(conv_1d)    

    bidirectional_simple_rnn_1 = Bidirectional(SimpleRNN(units, activation='relu', return_sequences=True, implementation=2, name='simple_rnn_1'))(batch_norm_cnn_1)
    batch_norm_bidirectional_simple_rnn_1 = BatchNormalization(name='batch_norm_bidirectional_simple_rnn_1')(bidirectional_simple_rnn_1)
    
    bidirectional_simple_rnn_2 = Bidirectional(SimpleRNN(units, activation='relu', return_sequences=True, implementation=2, name='simple_rnn_1'))(batch_norm_bidirectional_simple_rnn_1)
    batch_norm_bidirectional_simple_rnn_2 = BatchNormalization(name='batch_norm_bidirectional_simple_rnn_2')(bidirectional_simple_rnn_2)
    
    time_distributed_dense_1 = TimeDistributed(Dense(100), name='time_distributed_dense_1')(batch_norm_bidirectional_simple_rnn_2)
    batch_norm_time_distributed_dense_1 = BatchNormalization(name='batch_norm_time_distributed_dense_1')(time_distributed_dense_1)
    dropout_time_distributed_dense_1 = Dropout(.5, name='dropout_time_distributed_dense_1')(batch_norm_time_distributed_dense_1)

    time_distributed_dense_2 = TimeDistributed(Dense(50), name='time_distributed_dense_2')(dropout_time_distributed_dense_1)
    batch_norm_time_distributed_dense_2 = BatchNormalization(name='batch_norm_time_distributed_dense_2')(time_distributed_dense_2)
    dropout_time_distributed_dense_2 = Dropout(.5, name='dropout_time_distributed_dense_2')(batch_norm_time_distributed_dense_2)
    
    time_distributed_dense = TimeDistributed(Dense(output_dim), name='time_distributed_dense')(dropout_time_distributed_dense_2)
    y_pred = Activation('softmax', name='softmax')(time_distributed_dense)
    
    model = Model(inputs=input_data, outputs=y_pred)    
    model.output_length = lambda x: cnn_output_length_2(x, kernels, conv_border_mode, conv_stride)
    
    print(model.summary())
    
    return model