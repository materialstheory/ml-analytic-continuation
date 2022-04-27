import numpy as np
import tensorflow as tf
import sys, os, time, copy

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, Lambda, Conv1D, GaussianNoise, MaxPool1D, AveragePooling1D, Add, Dropout, Concatenate, BatchNormalization
import tensorflow.keras.backend as K

from scipy.linalg import svd

o2l = np.load("o2l_b40.npy")
u_, s_, vh_ = svd(o2l)
Lmax = o2l.shape[0]
u = tf.constant(u_.astype('float32'))
vh = tf.constant(vh_[:Lmax,:].astype('float32'))
s = tf.constant(np.diag(s_).astype('float32'))
omegac = 8
Nomega = 800
omega = np.linspace(-omegac, omegac, Nomega)
omega_t = tf.constant(omega.astype('float32'))
omega_t = tf.reshape(omega_t, (1, 1, Nomega))


# neural network 
def ResBlock(y, nodes, drop_ratio):
    l2_pen = 0
    x = Dense(nodes)(y)
    x = Dropout(drop_ratio)(x)
    x = Dense(nodes, activation = 'selu', kernel_regularizer = keras.regularizers.l2(l2 = l2_pen))(x)
    x = Dense(nodes)(x)
    y = Add()([y, x])
    
    return y

def StackResBlock(y, nodes, drop_ratio, layers):
    x = ResBlock(y, nodes, drop_ratio)
    for i in range(layers-1):
        x = ResBlock(x, nodes, drop_ratio)
    
    return x

def ResResBlocks(y, nodes, drop_ratio, layers, sublayers):
    
    for i in range(layers):
        x = StackResBlock(y, nodes, drop_ratio, sublayers)
        y = Add()([y, x])
    return y

def StackResResBlocks(y, nodes, drop_ratio, layers, sublayers, sub2layers):
    x = ResResBlocks(y, nodes, drop_ratio, sublayers, sub2layers)
    for i in range(layers-1):
        x = ResResBlocks(x, nodes, drop_ratio, sublayers, sub2layers)
        
    return x

def ResResResBlocks(y, nodes, drop_ratio, layers, sublayers, sub2layers, sub3layers):
    for i in range(layers):
        x = StackResResBlocks(y, nodes, drop_ratio, sublayers, sub2layers, sub3layers)
        y = Add()([y, x])

    return y


def A2G(A):
    """
    input: summation normalized spectral functions tensor
    output: Legendre coefficients of integral normalized spectral functions
    """
    G = tf.einsum('ij,nj->ni', vh, A)
    G = tf.einsum('ij,nj->ni', s, G)
    G = tf.einsum('ij,nj->ni', u, G)
    return Nomega/(2*omegac)*G

#y = tf.keras.layers.Lambda(lambda _: tf.random.normal(shape=(32,10), name="noise", dtype=tf.float64), dtype=tf.float64)(i)

def aug(A0, batch_size):
    
    
    wr = Lambda(lambda batch_size: tf.random.uniform(shape = (batch_size, 10, 1), minval = -0.8, maxval = 0.8)*omegac )(batch_size)
    
    #wr = tf.random.uniform(shape = (batch_size, 10, 1), minval = -0.8, maxval = 0.8)*8
    
    sigmar = Lambda(lambda batch_size: tf.random.uniform(shape = (batch_size, 10, 1), minval = 0.05, maxval = 0.2)*omegac )(batch_size)
    
    #sigmar = tf.random.uniform(shape = (batch_size, 10, 1), minval = 0.05, maxval = 0.2)*8
    
    cr = Lambda(lambda batch_size: tf.random.uniform(shape = (batch_size, 10, 1), minval = 0.001, maxval = 0.01))(batch_size)
    
    #cr = tf.random.uniform(shape = (batch_size, 10, 1), minval = 0.001, maxval = 0.01)
    
    
    dA = tf.math.exp(-0.5*(omega_t - wr)**2 / sigmar**2)/sigmar /np.sqrt(2*np.pi)*cr
    dA = tf.einsum('ijk->ik', dA)
    cr_sum = tf.einsum('ijk->ik', cr)
    
    A = (A0 + dA*(2*omegac)/Nomega)/(cr_sum + 1)
    
    return A
    


def custom_loss_function(x_true, out_pred):
    A_pred = out_pred[:, :Nomega]
    A_true = out_pred[:,  Nomega:]
    
    MAE = tf.abs(A_true - A_pred)
    return tf.reduce_mean(MAE, axis=-1)

def custom_metric_function(x_true, out_pred):
    A_pred = out_pred[:, :Nomega]
    A_true = out_pred[:,  Nomega:]
    
    MAE = tf.abs(A_true - A_pred)
    return tf.reduce_mean(MAE, axis=-1)


def create_model_aug(noise_level, nodes):
    
    output_shape = Nomega
    
    inputs = keras.Input(shape=(Lmax,))
    x = Dense(nodes, input_shape = (Lmax,), activation='selu')((2*omegac)/Nomega*inputs)
    x = ResResResBlocks(x, nodes, 0.1, 2, 2, 2, 2)
    outputs = Dense(Nomega, activation='softmax')(x)
    
    model_G2A = keras.Model(inputs=inputs, outputs = outputs, name = 'G2A')
    
    
    inputs = keras.Input(shape=(Nomega,))
    
    symbolic_shape = K.shape(inputs)
    A = aug(inputs, symbolic_shape[0])
    G0 = A2G(A)
    G0 = GaussianNoise(stddev = noise_level)(G0)
    
    A1 = model_G2A(G0)
    G1 = A2G(A1)
    outputs = Concatenate(axis = 1)([A1, A])
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="res_cnn")
    
    return model 


    
def create_model(noise_level, nodes):
    
    output_shape = Nomega
    
    inputs = keras.Input(shape=(Lmax,))
    x = Dense(nodes, input_shape = (Lmax,), activation='selu')((2*omegac)/Nomega*inputs)
    x = ResResResBlocks(x, nodes, 0.1, 2, 2, 2, 2)
    outputs = Dense(Nomega, activation='softmax')(x)
    
    model_G2A = keras.Model(inputs=inputs, outputs = outputs, name = 'G2A')
    
    
    inputs = keras.Input(shape=(Nomega,))
    
    symbolic_shape = K.shape(inputs)
    A = inputs 
    G0 = A2G(A)
    G0 = GaussianNoise(stddev = noise_level)(G0)
    if K.learning_phase() == False:
        G0 = tf.random.normal(shape = tf.shape(G0), mean = 0, stddev = noise_level) + G0
    
    A1 = model_G2A(G0)
    G1 = A2G(A1)
    outputs = Concatenate(axis = 1)([A1, A])
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="res_cnn")
    
    return model    
    
    
def main(training_data, validation_data, noise_level, prefix):
    
    A_omega_train_aug = np.load(training_data)
    A_omega_val_aug = np.load(validation_data)
    
    A_omega_train_rep = np.copy(A_omega_train_aug)
    A_omega_val_rep = np.copy(A_omega_val_aug)
    
    
    val_shape = A_omega_val_aug.shape[0]
    
    # make small variation in spectral functions: one copy for validation, five for training
    for i in range(9):        
        A_omega_val_rep[int(val_shape/9*i):int(val_shape/9*(i+1))] = aug(
            tf.constant(A_omega_val_aug[int(val_shape/9*i):int(val_shape/9*(i+1))].astype('float32')),
                              int(val_shape/9)).numpy()
    
    train_shape = A_omega_train_aug.shape[0]
    
    for j in range(5):        
        for i in range(9):
            
            A_omega_train_rep = np.concatenate([A_omega_train_rep, 
                                     aug(tf.constant(A_omega_train_aug[int(train_shape/9*i):int(train_shape/9 * (i+1))].astype('float32')), 
                                               int(train_shape/9)).numpy()], axis = 0)
    
    model = create_model(np.power(10.0, -noise_level), 100)
    
    
    label = str(prefix) + "_mult5-res2222_100-n" + str(noise_level)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001/2, epsilon = 1e-12),
              loss = custom_loss_function,
              metrics=[custom_metric_function])
    
    callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/1.2,
                              patience=40, min_lr=1e-6/2,verbose=1, cooldown = 30)

    cp_best_callback = tf.keras.callbacks.ModelCheckpoint(filepath="cp_"+label+"_best",
                                                          save_freq='epoch',
                                                          save_weights_only=True,
                                                          verbose = 0,
                                                          save_best_only = True,
                                                          monitor='val_loss')
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="cp_"+label+"_test",
                                                     save_freq='epoch',
                                                     save_weights_only=True,
                                                     verbose=1)
    
    log_callback = tf.keras.callbacks.CSVLogger("log_cp_"+label+".csv", append=True, separator=';')


    history = model.fit(A_omega_train_rep, A_omega_train_rep, shuffle = True, batch_size = 400, epochs = 1600, steps_per_epoch = 693,
                    validation_data=(A_omega_val_rep, A_omega_val_rep), callbacks=[callback, cp_callback, cp_best_callback, log_callback])
    
    
    #train_loss = history.history['loss']
    #val_loss = history.history['val_loss']
    
    return 0
    
    #np.save("train_loss_noaug.npy", train_loss)
    #np.save("val_loss_noaug.npy", val_loss)


if __name__ == "__main__":
    print("running .py...")
    training_data= str(sys.argv[1])  # file name e.g. "A_omega_train.npy" "A_easy_omega_train.npy"
    validation_data = str(sys.argv[2]) # file name e.g. "A_omega_val.npy" "A_easy_omega_val.npy"
    noise_level = int(sys.argv[3])  # negative exponent, 2, 3, 4
    prefix = str(sys.argv[4]) # "Arsenault" or "easy"

    main(training_data, validation_data, noise_level, prefix)
