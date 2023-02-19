import tensorflow as tf
# import tensorflow_addons as tfa


def save_as_tflite(model, name='model'):
    model.input.set_shape(1 + model.input.shape[1:])
    model.save(name + '.h5')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(name + '.tflite', 'wb').write(tflite_model)

    # to json
    model_json = model.to_json()
    open(name + '.json', 'w').write(model_json)


class MyModel():
    def  __init__(self,
             model_name='bwunet',
             input_shape=(128, 128, 3),
             use_bn=True,
             kernel_size=3,
             kernel_regularizer=None,
             kernel_constraint=6,
             ):

        self.input_shape = input_shape
        self.use_bn = use_bn
        self.kernel_size = kernel_size

        self.kernel_regularizer = None
        if kernel_regularizer != None:
            self.kernel_regularizer = tf.keras.regularizers.L2(l2=0.01)

        self.kernel_constraint = None
        if kernel_constraint != None:
            self.kernel_constraint = tf.keras.constraints.MinMaxNorm(
                min_value=0.0, max_value=kernel_constraint, rate=1, axis=[0, 1, 2])
            self.bias_constraint = tf.keras.constraints.MinMaxNorm(
                min_value=0.0, max_value=kernel_constraint, rate=1, axis=0)

    def _myactivation_layer(self, activation='relu'):
        if activation.lower() == 'relu':
            activation_layer = tf.keras.layers.ReLU()
        elif activation.lower() == 'leakyrelu':
            activation_layer = tf.keras.layers.LeakyReLU()
        else:
            activation_layer = tf.keras.activations.linear()
        return activation_layer

    def _mynorm(self, x, type, nblock, num):
        out = self._mynorm_layer(type)(x)
        return out

    def _mynorm_layer(self, type=None, name=None):
        if not isinstance(type, str):
            out = tf.keras.activations.linear
        elif type.lower() == 'None':
            out = tf.keras.activations.linear
        else:
            out = tf.keras.layers.BatchNormalization(name=name)
        return out

    def conv2d(self, filters, kernel_size=3, strides=1, padding='same',
                kernel_initializer='glorot_uniform', bias_initializer='zeros', use_bias=True,  activation=None, name=None):
        return tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, use_bias=use_bias,
                                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer, bias_regularizer=None,
                                    kernel_constraint=self.kernel_constraint, bias_constraint=self.bias_constraint,
                                    padding=padding, activation=activation, name=name)

    def tconv2d(self, filters, kernel_size=3, strides=2, padding='same',
                kernel_initializer='glorot_uniform', bias_initializer='zeros', use_bias=True,  activation=None, name=None):
        return tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,use_bias=use_bias,
                                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                            kernel_regularizer=self.kernel_regularizer, bias_regularizer=None,
                                            kernel_constraint=self.kernel_constraint, bias_constraint=self.bias_constraint,
                                            padding=padding, activation=activation, name=name)


    def _enc_block(self, filters:int, kernek_size:int=3, strides:int=1, padding:str='same', norm:str='None', name=None):
        def func(x):
            enc = self.conv2d(filters, kernek_size, strides,
                              padding, activation=None, name=name)(x)
            enc = self._mynorm_layer(norm)(enc)
            enca = tf.keras.layers.PReLU(shared_axes=[1, 2], name=name + '_prelu')(enc)
            out = tf.keras.layers.Add()([enca, x])
            return out
        return func


    def get_model_mc(self):


        input = tf.keras.layers.Input(shape=self.input_shape, name='mc_input')

        # encoder
        enc0  = self.conv2d(filters=32, kernel_size=self.kernel_size, name='enc0')(input)
        enc0a = tf.keras.layers.PReLU(shared_axes=[1, 2], name='enc0_prelu')(enc0)

        enc1 = self._enc_block(filters=32, name='enc1')(enc0a)
        enc2 = self._enc_block(filters=32, name='enc2')(enc1)

        enc3  = self.conv2d(filters=32, kernel_size=self.kernel_size, strides=2, name='enc3')(enc2)  # ds1
        enc3a = tf.keras.layers.PReLU(shared_axes=[1, 2], name='enc3_prelu')(enc3)

        enc4 = self._enc_block(filters=32, name='enc4')(enc3a)

        enc5 = self.conv2d(filters=32, kernel_size=self.kernel_size, strides=2, name='enc5')(enc4)  # ds2
        enc5a = tf.keras.layers.PReLU(shared_axes=[1, 2], name='enc5_prelu')(enc5)

        enc6 = self._enc_block(filters=32, name='enc6')(enc5a)
        enc7 = self._enc_block(filters=32, name='enc7')(enc6)

        # decoder
        dec6 = self.tconv2d(filters=32, kernel_size=self.kernel_size, strides=2, name='dec6_tconv')(enc7)
        dec6a = tf.keras.layers.PReLU(shared_axes=[1, 2], name='dec6_prelu')(dec6)

        dec6a_add = tf.keras.layers.Add()([dec6a, enc4])


        dec6e = self._enc_block(filters=32, name='dec6e')(dec6a_add)

        dec3 = self.tconv2d(filters=32, kernel_size=self.kernel_size, strides=2, name='dec3_tconv')(dec6e)
        dec3a = tf.keras.layers.PReLU(shared_axes=[1, 2], name='dec3_prelu')(dec3)

        dec3a_add = tf.keras.layers.Add()([dec3a, enc2])

        dec2e = self._enc_block(filters=32, name='dec2e')(dec3a_add)
        dec1e = self._enc_block(filters=32, name='dec1e')(dec2e)
        dec0e = self._enc_block(filters=32, name='dec0e')(dec1e)

        out = self.tconv2d(filters=16, kernel_size=self.kernel_size, strides=2, name='out_tconv')(dec0e)

        model = tf.keras.Model(inputs=input, outputs=out, name='MC')
        # model = tf.keras.Model(inputs=input, outputs=enc0a, name='MC')

        return model

    def get_model_rgb(self):

        input = tf.keras.layers.Input(shape=self.input_shape, name='mc_input')

        # encoder
        enc0 = self.conv2d(filters=32, kernel_size=self.kernel_size, name='enc0')(input)
        enc0a = tf.keras.layers.PReLU(shared_axes=[1, 2], name='enc0_prelu')(enc0)

        print('enc0.dtype %s' % enc0.dtype.name)
        print('enc0a.dtype %s' % enc0a.dtype.name)
        # exit()

        enc1 = self._enc_block(filters=32, name='enc1')(enc0a)
        enc2 = self._enc_block(filters=32, name='enc2')(enc1)

        enc3 = self.conv2d(filters=32, kernel_size=self.kernel_size, strides=2, name='enc3')(enc2)  # ds1
        enc3a = tf.keras.layers.PReLU(shared_axes=[1, 2], name='enc3_prelu')(enc3)

        enc4 = self._enc_block(filters=32, name='enc4')(enc3a)

        enc5 = self.conv2d(filters=32, kernel_size=self.kernel_size, strides=2, name='enc5')(enc4)  # ds2
        enc5a = tf.keras.layers.PReLU(shared_axes=[1, 2], name='enc5_prelu')(enc5)

        enc6 = self._enc_block(filters=32, name='enc6')(enc5a)
        enc7 = self._enc_block(filters=32, name='enc7')(enc6)

        # decoder
        dec6 = self.tconv2d(filters=32, kernel_size=self.kernel_size, strides=2, name='dec6_tconv')(enc7)
        dec6a = tf.keras.layers.PReLU(shared_axes=[1, 2], name='dec6_prelu')(dec6)

        dec6a_add = tf.keras.layers.Add()([dec6a, enc4])

        dec6e = self._enc_block(filters=32, name='dec6e')(dec6a_add)

        dec3 = self.tconv2d(filters=32, kernel_size=self.kernel_size, strides=2, name='dec3_tconv')(dec6e)
        dec3a = tf.keras.layers.PReLU(shared_axes=[1, 2], name='dec3_prelu')(dec3)

        dec3a_add = tf.keras.layers.Add()([dec3a, enc2])

        dec2e = self._enc_block(filters=32, name='dec2e')(dec3a_add)
        dec1e = self._enc_block(filters=32, name='dec1e')(dec2e)

        # output
        out = self.conv2d(filters=3, kernel_size=self.kernel_size, strides=1,name='out')(dec1e)
        print('out.dtype %s' % out.dtype.name)

        # output activation should be 32bits
        out = tf.keras.layers.Activation('tanh', dtype='float32')(out)
        print('out.dtype %s' % out.dtype.name)
        # exit()


        model = tf.keras.Model(inputs=input, outputs=out, name='RGB')

        # model = tf.keras.Model(inputs=input, outputs=enc0a, name='MC')

        return model


def main():
    print(tf.__version__)

    mymodel = MyModel()
    # model = mymodel.get_model_mc()
    model = mymodel.get_model_rgb()
    model.summary()

    save_as_tflite(model)

    print("main done done")
if __name__ == "__main__":
    main()
