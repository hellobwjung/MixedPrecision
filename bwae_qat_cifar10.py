
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse

import os
import glob
import datetime
import numpy as np

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import keras
import tensorflow_model_optimization as tfmot

import matplotlib.pyplot as plt
from PIL import Image
from skimage import io

from random import shuffle

from IPython.display import clear_output
clear_output(wait=False)


# LAERNING RATE
LEARNING_RATE = 1e-4

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

NGPU = len(get_available_gpus())
if NGPU == 0:
    NGPU = 1

MODEL_NAME = __file__.split('.')[0]  # 'model_tetra_out_model_tetra_12ch'





def plot_gt_and_predictions(model, test_images, is_plot=True, input_max=1., path='plot.png'):
    predictions = model.predict(test_images[:16,...])
    predictions = np.clip(predictions, 0, input_max)

    if is_plot:
        fig = plt.figure(figsize=(16, 16))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 8, 2*i + 1)
            plt.imshow(test_images[i, :, :, :] )
            plt.axis('off')

            plt.subplot(4, 8, 2*i + 2)
            plt.imshow(predictions[i, :, :, :])
            plt.axis('off')

        plt.show()
        plt.savefig(path)

    return predictions


def plot_two_predictions(images1, images2, input_max=1.):
    images1 = np.clip(images1, 0, input_max)
    images2 = np.clip(images2, 0, input_max)


    fig = plt.figure(figsize=(16, 16))
    for i in range(images1.shape[0]):
        plt.subplot(4, 8, 2*i + 1)
        plt.imshow(images1[i, :, :, :], vmin=0, vmax=input_max )
        plt.axis('off')
        plt.subplot(4, 8, 2*i + 2)
        plt.imshow(images2[i, :, :, :], vmin=0, vmax=input_max)
        plt.axis('off')

    plt.show()

    return None


def generate_and_save_images(model, epoch, test_images, input_max=1.):
    predictions = model.predict(test_images[:16,...])
    predictions = np.clip(predictions, 0, input_max)


    fig = plt.figure(figsize=(16, 16))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 8, 2*i + 1)
        plt.imshow(test_images[i, :, :, :] , vmin=0, vmax=input_max)
        plt.axis('off')

        plt.subplot(4, 8, 2*i + 2)
        plt.imshow(predictions[i, :, :, :], vmin=0, vmax=input_max)
        plt.axis('off')



    plt.show()

    return predictions

def plot_images(images, input_max=1.):
    images = images[:16,...]
    images = np.clip(images, 0, input_max)


    fig = plt.figure(figsize=(8, 8))
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i + 1)
#         im = plt.imshow(images[i, :, :, :].astype(np.uint8))
        im = plt.imshow(images[i, :, :, :])
        plt.axis('off')

    plt.show()




def plot_multi_images(images, input_max=1.): # images: list
    num_sets = len(images)
    print('--------------num_Sets', num_sets)

    images = np.clip(images, 0, input_max)
    images = images*255
    images = images.astype(np.uint8)

    fig, axes = plt.subplots(nrows=8, ncols=2*num_sets, figsize=(16, 16))
    cnt=0
    for ax in axes.flat:

        im = ax.imshow(images[cnt%num_sets][cnt//num_sets, :, :, :], vmin=0, vmax=255 )

        ax.set_xticks([])
        ax.set_yticks([])
        cnt+=1
#     fig.colorbar(im, ax=axes.ravel().tolist())

    plt.show()




def plot_images_with_color_bar(images, input_max=1.):
    images = images[:16,...]
    images = np.clip(images, 0, input_max)
    images = images*255
    images = images.astype(np.uint8)

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))
    cnt=0
    for ax in axes.flat:
        im = ax.imshow(images[cnt, :, :, :], vmin=0, vmax=input_max, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

        cnt+=1

    fig.colorbar(im, ax=axes.ravel().tolist())

    plt.show()


## Quantization configs
LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer

class DefaultDenseQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def __init__(self) -> None:
        super().__init__()
        self.a_bits = 8
        self.w_bits = 8

    def __init__(self, a_bits=8, w_bits=8) -> None:
        super().__init__()
        self.a_bits = a_bits
        self.w_bits = w_bits


    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
      return [(layer.kernel, LastValueQuantizer(num_bits=self.w_bits, symmetric=True, narrow_range=False, per_axis=True))]

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
      return [(layer.activation, MovingAverageQuantizer(num_bits=self.a_bits, symmetric=False, narrow_range=False, per_axis=False))]

    def set_quantize_weights(self, layer, quantize_weights):
      # Add this line for each item returned in `get_weights_and_quantizers`
      # , in the same order
      layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
      # Add this line for each item returned in `get_activations_and_quantizers`
      # , in the same order.
      layer.activation = quantize_activations[0]

    # Configure how to quantize outputs (may be equivalent to activations).
#     def get_output_quantizers(self, layer):
#       return []
    def get_output_quantizers(self, layer):
        return [MovingAverageQuantizer(
            num_bits=self.a_bits,
            per_axis=False, symmetric=False, narrow_range=False)]

    def get_config(self):
      return {}


class ModifiedDenseQuantizeConfig(DefaultDenseQuantizeConfig):

    def __init__(self) -> None:
        super().__init__()
        self.a_bits = 8

    def __init__(self, a_bits=8) -> None:
        super().__init__(a_bits, w_bits=8)

    def __init__(self, a_bits=8, w_bits=8) -> None:
        super().__init__(a_bits, w_bits)
        # self.a_bits = a_bits
        # self.w_bits = w_bits

    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        return []

    def set_quantize_activations(self, layer, quantize_activations):
        return []

    # Configure weights to quantize with 4-bit instead of 8-bits.
    def get_output_quantizers(self, layer):
        # return [MovingAverageQuantizer(
        #     num_bits=8, per_axis=False, symmetric=False, narrow_range=False)]
            return [MovingAverageQuantizer(
        num_bits=self.a_bits,
        per_axis=False, symmetric=False, narrow_range=False)]





## DEFINE MODEL
def get_model(name='bwae', patch_size=32):

    ch1, ch2, ch3, ch4 = 16, 32, 64, 128

    connections = [True, True, True]
    channels = [ch3, ch2, ch1]

    iinput = tf.keras.layers.Input((patch_size, patch_size, 3), name="input")

    x1 = tf.keras.layers.Conv2D(ch1, 3, strides=1, activation="relu", padding="same", name="enc1" )(iinput)
    x2 = tf.keras.layers.Conv2D(ch2, 3, strides=2, activation="relu", padding="same", name="enc2" )(x1)
    x3 = tf.keras.layers.Conv2D(ch3, 3, strides=2, activation="relu", padding="same", name="enc3" )(x2)
    x4 = tf.keras.layers.Conv2D(ch4, 3, strides=2, activation="relu", padding="same", name="enc4" )(x3)
    x = x4

    lrsc = [x3, x2, x1]
    for i in range(len(connections)):
        namet = 'dec%d' %i
        namec = 'dec%d_conv' % i
        name_a = 'add%d' % i

        activationt = None if connections[i] else "relu"

        x = tf.keras.layers.Conv2DTranspose(channels[i], 3, strides=2, activation=activationt, padding="same", name=namet)(x)

        if connections[i]:
            x = tf.keras.layers.Add(name=name_a)([x, lrsc[i] ])
        x = tf.keras.layers.Conv2D(ch3, 3, strides=1, padding="same", name=namec, activation="relu")(x)

    x = tf.keras.layers.Conv2D(3, 3, strides=1, activation="relu", padding="same", name="last_conv")(x)

    model = tf.keras.models.Model(iinput, x, name=name)

    return model


def get_Q_model(name='bwae_q', a_bits=8, w_bits=8, patch_size=32):
    quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer

    ch1, ch2, ch3, ch4 = 16, 32, 64, 128

    connections = [True, True, True]
    channels = [ch3, ch2, ch1]


    iinput = tf.keras.layers.Input((patch_size, patch_size, 3), name="input")
    x1 = quantize_annotate_layer(tf.keras.layers.Conv2D(ch1, 3, strides=1, activation="relu", padding="same", name="enc1" ),
                                       DefaultDenseQuantizeConfig(a_bits=a_bits, w_bits=w_bits))(iinput)
    x2 = quantize_annotate_layer(tf.keras.layers.Conv2D(ch2, 3, strides=2, activation="relu", padding="same", name="enc2" ),
                                       DefaultDenseQuantizeConfig(a_bits=a_bits, w_bits=w_bits))(x1)
    x3 = quantize_annotate_layer(tf.keras.layers.Conv2D(ch3, 3, strides=2, activation="relu", padding="same", name="enc3" ),
                                       DefaultDenseQuantizeConfig(a_bits=a_bits, w_bits=w_bits))(x2)
    x4 = quantize_annotate_layer(tf.keras.layers.Conv2D(ch4, 3, strides=2, activation="relu", padding="same", name="enc4" ),
                                       DefaultDenseQuantizeConfig(a_bits=a_bits, w_bits=w_bits))(x3)
    x = x4

    lrsc = [x3, x2, x1]
    for i in range(len(connections)):
        namet = 'dec%d' %i
        namec = 'dec%d_conv' % i
        name_a = 'add%d' % i

        activationt = None if connections[i] else "relu"

        x = quantize_annotate_layer(
                tf.keras.layers.Conv2DTranspose(channels[i], 3, strides=2, activation=activationt, padding="same", name=namet),
                        DefaultDenseQuantizeConfig(a_bits=a_bits, w_bits=w_bits))(x)

        if connections[i]:
            x = quantize_annotate_layer(tf.keras.layers.Add(name=name_a),
                                            ModifiedDenseQuantizeConfig(a_bits=a_bits))([x, lrsc[i] ])

        x = quantize_annotate_layer(tf.keras.layers.Conv2D(ch3, 3, strides=1, padding="same", name=namec, activation="relu"),
                                        DefaultDenseQuantizeConfig(a_bits=a_bits, w_bits=w_bits))(x)
    x = quantize_annotate_layer(tf.keras.layers.Conv2D(3, 3, strides=1, activation="relu", padding="same", name="last_conv"),
                                       DefaultDenseQuantizeConfig(a_bits=a_bits, w_bits=w_bits))(x)

    model = tf.keras.models.Model(iinput, x, name=name)

    return model


def get_tflite_model(tfname):
    interpreter = tf.lite.Interpreter( model_path=tfname)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    ifloating_model = input_details[0]['dtype'] == np.float32
    print('ifloating_model: ', ifloating_model)

    ofloating_model = output_details[0]['dtype'] == np.float32
    print('ofloating_model: ', ofloating_model)

    return interpreter


import time
def inference_one_tflite(tfname, input_files, postfix='',output_path='outputs', input_max=1.):
    opath = os.path.join(output_path, tfname)
    os.makedirs(opath, exist_ok=True)

    interpreter = get_tflite_model(tfname)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()




    for idx, arr in enumerate(input_files):
    # for idx, f in enumerate(range(8)):
    #
        fname = os.path.join(opath, "org%04d.png" % (idx))
        io.imsave(fname, arr)


        arr_input = arr # [0, 1]
        arr_input = arr_input[None,...]


        interpreter.set_tensor(input_details[0]['index'], arr_input)

        start_time = time.time()
        interpreter.invoke()
        stop_time = time.time()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)
        results = np.clip(results, 0, 1)

        #
        print(idx, 'time: {:.3f}ms'.format((stop_time - start_time) * 1000))


        name = os.path.join(opath, "inf%04d%s.png" % (idx, postfix))
        io.imsave(name, results)


    pass




def main(args):
    print(tf.__version__)


    # retrieve args
    # update params. using input arguments

    patch_size = args.patch_size
    batch_size = args.batch_size
    myepoch = args.epoch
    input_max = args.input_max

    a_bits = args.a_bits
    w_bits = args.w_bits

    ## LOAD CIFAR10
    cifar10 = keras.datasets.cifar10

    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_images = (train_images / 255.0) * input_max
    test_images = (test_images / 255.0) * input_max

    print(type(train_images[0,0,0,0]))

    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)
    print(type(train_images[0,0,0,0]))
    print(np.amin(test_images),np.amax(test_images))


    plt.subplot(141), plt.imshow(train_images[0], interpolation="bicubic"), plt.grid(False)
    plt.subplot(142), plt.imshow(train_images[4], interpolation="bicubic"), plt.grid(False)
    plt.subplot(143), plt.imshow(train_images[8], interpolation="bicubic"), plt.grid(False)
    plt.subplot(144), plt.imshow(train_images[12], interpolation="bicubic"), plt.grid(False)
    plt.show()


    # BASE_DIR
    base_dir = 'model_bwae'
    base_dir = '.'
    os.makedirs(base_dir, exist_ok=True)

    ## DEFINE FLOAT MODEL
    # define model
    model = get_model(patch_size=patch_size)

    model.save(os.path.join(base_dir, 'bwae.h5'))
    model.summary()
    tf.keras.utils.plot_model(model, to_file=os.path.join(base_dir, 'bwae_model_float.png'), show_shapes=True, dpi=64)


    # predict before train
    opath = os.path.join(base_dir, 'bwae_float_before.png')
    before = plot_gt_and_predictions(model, test_images[:16,...], path=opath)

    # train the float model
    optimizer_f = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, name='Adam_f')
    model.compile(optimizer=optimizer_f, loss=["mse"])


    history = model.fit(train_images,
                        train_images,
                        batch_size=batch_size,
                        epochs=myepoch,
                        validation_data=(test_images, test_images))

    model.save(os.path.join(base_dir, 'bwae_1.h5'))


    # predict before train
    opath = os.path.join(base_dir, 'bwae_float_after.png')
    after = plot_gt_and_predictions(model, test_images[:16,...], path=opath)



    ## DEFINE QUANTIZED MODEL
    # define quantization annotated model & load weights from float model
    model_q = get_Q_model(a_bits=a_bits, w_bits=w_bits)

    model_q.load_weights(os.path.join(base_dir, 'bwae.h5'))
    model_q.save(os.path.join(base_dir, 'bwae_1_q.h5'))
    model_q.summary()
    tf.keras.utils.plot_model(model_q, to_file=os.path.join(base_dir, 'bwae_model_Qannotated.png'), show_shapes=True, dpi=64)

    # predict before train
    opath = os.path.join(base_dir, 'bwae_qat1_before.png')
    before = plot_gt_and_predictions(model_q, test_images[:16,...], path=opath)


    # make quantized model using quantize_apply
    quantize_model   = tfmot.quantization.keras.quantize_model
    quantize_scope = tfmot.quantization.keras.quantize_scope
    quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model


    with quantize_scope( {'DefaultDenseQuantizeConfig': DefaultDenseQuantizeConfig,
                        'ModifiedDenseQuantizeConfig': ModifiedDenseQuantizeConfig} ):
        q_aware_model = tfmot.quantization.keras.quantize_apply(model_q)


    q_aware_model.summary()
    tf.keras.utils.plot_model(q_aware_model, to_file=os.path.join(base_dir, 'bwae_model_quantized.png'), show_shapes=True, dpi=64)

    # train the QAT model
    optimizer_q = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, name='Adam_f')
    q_aware_model.compile(optimizer=optimizer_q, loss=["mse"])

    history = q_aware_model.fit(train_images,
                                train_images,
                                batch_size=batch_size,
                                epochs=myepoch,
                                validation_data=(test_images, test_images))
    q_aware_model.save(os.path.join(base_dir, 'bwae_1_qat_1.h5'))

    # predict after train
    opath = os.path.join(base_dir, 'bwae_qat1_after.png')
    after = plot_gt_and_predictions(q_aware_model, test_images[:16,...], path=opath)



    ## FREEZE WEIGHTS
    # float
    print('-----------> freeze start tflite FLOAT')
    input_name = model.input_names[0]
    index = model.input_names.index(input_name)
    model.inputs[index].set_shape([1, patch_size, patch_size, 3])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float_model = converter.convert()
    fname = os.path.join(base_dir, "bwae_1.tflite")
    open(fname, 'wb').write(tflite_float_model)
    print('<----------- freeze done tflite FLOAT')


    # qat
    print('-----------> freeze start tflite QAT')
    input_name = q_aware_model.input_names[0]
    index = q_aware_model.input_names.index(input_name)
    q_aware_model.inputs[index].set_shape([1, patch_size, patch_size, 3])


    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_qat_model = converter.convert()
    fname = os.path.join(base_dir, "bwae_1_qat_A%dW%d.tflite" %(a_bits, w_bits))
    open(fname, 'wb').write(tflite_qat_model)



    print('<----------- freeze done tflite QAT')


    ## inference with tflite


    fname_float = os.path.join(base_dir, "bwae_1.tflite")
    fname_qat = os.path.join(base_dir, "bwae_1_qat_A%dW%d.tflite" %(a_bits, w_bits))

    inference_one_tflite(fname_float, test_images[:16,...], postfix='_float',output_path='outputs', input_max=1.)
    inference_one_tflite(fname_qat, test_images[:16,...], postfix='_qat',output_path='outputs', input_max=1.)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--patch_size',
        type=int,
        default=32,
        help='input patch size')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='input patch size')

    parser.add_argument(
        '--epoch',
        type=int,
        default=1,
        help='epoch')

    parser.add_argument(
        '--input_max',
        type=float,
        default=1,
        help='input_max')

    parser.add_argument(
        '--a_bits',
        type=int,
        default=16,
        help='# of activation bits')

    parser.add_argument(
        '--w_bits',
        type=int,
        default=8,
        help='# of weight bits')
    args = parser.parse_args()

    main(args)
