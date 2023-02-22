import os
import argparse
import time
import glob
import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf

from skimage import io
from get_image_from_tfrecords import get_image


def get_index(cfa_pattern=2, crop_size=128):
    idx_R = np.tile(
        np.concatenate(
            (np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1),
             np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)),
            axis=0),
        (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))


    idx_B = np.tile(
        np.concatenate(
            (np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
             np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)),
            axis=0),
        (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

    idx_G = np.tile(
        np.concatenate(
            (np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
             np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1)),
            axis=0),
        (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

    idx_RGB = np.concatenate((idx_R[..., np.newaxis],
                              idx_G[..., np.newaxis],
                              idx_B[..., np.newaxis]), axis=-1)
    return idx_RGB










def get_file_list(dataset_path = 'N:/dataset/MIT/imgs/images_sub10/valtest'):
    dataset_path = '/home/dataset/MIT/imgs/images_sub10/valtest'
    files = glob.glob(os.path.join(dataset_path, '**', '*.png'), recursive=True)
    files.sort()
    # for idx, f in enumerate(files):
    #     print(idx+1," / ", len(files), ', ', f )
    #     if idx>10:
    #         return
    return files



def load_checkpoint_if_exists(model, model_dir, model_name):
    prev_epoch = 0

    prev_loss = np.inf

    p = os.path.join(model_dir, '*.h5' )
    print('p =', p)
    trained_weights = glob.glob(p)
    print('--=-=-=-> ', trained_weights)
    if len(trained_weights ) > 0:
        print('===========> %d TRAINED WEIGHTS EXIST' % len(trained_weights))
        trained_weights.sort()
        trained_weights = trained_weights[-1]
        print('---------------------> ', trained_weights)
        model.load_weights(trained_weights)
        # idx = trained_weights.rfind(model_name)
        # prev_epoch = int(trained_weights[idx-6:idx-1])
        # prev_loss = float(trained_weights.split('_')[-1][:-3])
        trained_weights.replace('\\\\', '/')
        wname = trained_weights.split(model_name)[-2]
        print('wname, ', wname, ',', wname[-6:-1])
        prev_epoch = int(wname[-6:-1])
        print('prev epoch', prev_epoch)
    else:
        print('===========> TRAINED WEIGHTS NOT EXIST', len(trained_weights))

    return model, prev_epoch, trained_weights



def get_structure(data_path, model_name, model_sig):
    print(os.getcwd())
    if ':' in os.getcwd():
        print('this is windows, ', os.getcwd())
        data_path = 'weights'
        # exit()

    # load model structure
    bittage = 16 if '16' in model_sig else 32
    path = os.path.join(data_path, f'*{model_name}*{model_sig}*.h5')
    h5s = glob.glob(path)
    if len(h5s)<1:
        print('no structure files, h5s=', h5s)
        print('path=', path)
        exit()

    print(h5s, path)
    h5 = h5s[0]
    # h5 = os.path.join(data_path,'MC_rmsc_model_structure.h5')
    print(h5s, h5, path)
    model = tf.keras.models.load_model(h5, custom_objects={'tf': tf})


    return model




def get_model(data_path, model_name, model_sig):

    print('get_model -> get_structure <-------------------------------')
    model = get_structure(data_path, model_name, model_sig)

    # load weights
    p = os.path.join(data_path, model_name + model_sig)
    print(p)
    model, prev_epoch, h5name = load_checkpoint_if_exists(model, p, model_name)
    # model, prev_epoch, h5name = model, 0, 'babo'

    return model, prev_epoch, h5name



def freeze_model(model, fname):
    model.input.set_shape(1 + model.input.shape[1:])  # to freeze model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(fname , "wb").write(tflite_model)

def freeze(args):
    ## get h5 & make it tflite
    data_path = args.data_path
    model_name = args.model_name
    model_sig = args.model_sig
    test = args.test

    model_sig16 = '_16bit'
    model_sig32 = '_32bit'
    model_sig16 = '_16bit_2_same_noise'
    model_sig32 = '_32bit_2_same_noise'
    # model_sig16 = '_16bit_4_same_noise'
    # model_sig32 = '_32bit_4_same_noise'

    structure16 = get_structure(data_path, model_name, model_sig32) # <--load 32, important!!!
    # structure32 = get_structure(data_path, model_name, model_sig32)

    model16, _, h5name16 = get_model(data_path, model_name, model_sig16)
    model32, _, h5name32 = get_model(data_path, model_name, model_sig32)

    weights32 = model32.get_weights()
    weights16 = model16.get_weights()
    model16 = structure16
    model16.set_weights(weights16)

    print(len(weights16))


    # freeze model
    tfname16 = h5name16[:-3] + '.tflite'
    tfname32 = h5name32[:-3] + '.tflite'
    print('fname16: ', tfname16)
    print('fname32: ', tfname32)
    if test==False:
        freeze_model(model16, tfname16)
        freeze_model(model32, tfname32)

    return tfname16, tfname32


def get_tflite_model(tfname):
    interpreter = tf.lite.Interpreter(
        model_path=tfname)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32
    print('floating_model: ', floating_model)

    return interpreter

def inference_one_tflite(tfname, input_files, output_path='outputs'):
    opath = os.path.join(output_path, tfname)
    os.makedirs(opath, exist_ok=True)

    interpreter = get_tflite_model(tfname)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    idx_RGB = get_index()

    for idx, f in enumerate(input_files):
    # for idx, f in enumerate(range(8)):
        img = Image.open(f)
        arr = np.asarray(img)

        # arr = get_image()

        arr_input = (arr * idx_RGB).astype(np.float32) # [0, 255]
        patternized = (arr_input / 127.5) - 1 # [0, 255] --> # [0, 2] --> # [-1, 1]
        # patternized = (arr_input * idx_RGB).astype(np.float32)
        patternized = patternized[None,...]
        print(arr.shape, idx_RGB.shape, patternized.shape, patternized.dtype)


        img.save(os.path.join(opath, "org%04d.png" % (idx)))

        p2 = ((patternized+1)*127.5).astype(np.uint8)
        img = Image.fromarray(p2[0])
        name = os.path.join(opath, "in%04d.png" % (idx))
        print(name)
        img.save(name)


        interpreter.set_tensor(input_details[0]['index'], patternized)

        start_time = time.time()
        interpreter.invoke()
        stop_time = time.time()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)

        # print('result.shape', results.shape, np.amin(results), np.amax(results))
        #
        print(idx, 'time: {:.3f}ms'.format((stop_time - start_time) * 1000))

        # results = ((results+1)* 127.5) + 0.5
        # results = np.clip(results, 0, 255).astype(np.uint8)
        # img = Image.fromarray(results)
        # name = os.path.join(opath, "%04d.png"%(idx))
        # print(name)
        # img.save(name)

        results = (results + 1) / 2
        name = os.path.join(opath, "%04d.png" % (idx))
        io.imsave(name, results)




        # if idx>9:
        #     return



    pass


def inference_tflite(args, tfnames):
    # tfname16 = os.path.join('weights', 'MC_rmsc_16bit', '00598_MC_rmsc_16bit_3.08692e+02.tflite')
    # tfname32 = os.path.join('weights', 'MC_rmsc_32bit', '00589_MC_rmsc_32bit_3.09384e+02.tflite')
    tfname16, tfname32 = tfnames
    print('fname16: ', tfname16)
    print('fname32: ', tfname32)

    fname = os.path.join('foveon',
                         'tetra',
                         '04_SDQuattroH_PotableBox_1000lx_5000K_1By30s_ISO100_F5.6_0_tetra.npy')
    foveon = np.load(fname)
    print(foveon.shape, foveon.dtype)


    input_files = get_file_list()

    for idx, tfname in enumerate(tfnames):

        inference_one_tflite(tfname, input_files)




def inference_h5(args):
    ## get h5
    data_path = args.data_path
    model_name = args.model_name
    test = args.test

    model_sig16 = '_16bit'
    model_sig32 = '_32bit'
    model_sig16 = '_16bit_2_same_noise'
    model_sig32 = '_32bit_2_same_noise'
    model_sig16 = '_16bit_4_same_noise'
    model_sig32 = '_32bit_4_same_noise'

    structure16 = get_structure(data_path, model_name, model_sig32)  # <--load 32, important!!!
    structure32 = get_structure(data_path, model_name, model_sig32)


    model16, _, h5name16 = get_model(data_path, model_name, model_sig16)
    model32, _, h5name32 = get_model(data_path, model_name, model_sig32)

    weights16 = model16.get_weights()
    # weights32 = model32.get_weights()
    # model16 = structure16
    # model32 = structure32
    model16.set_weights(weights16)

    opath16 = 'outputs/weights/MC_rmsc_16bit'
    opath32 = 'outputs/weights/MC_rmsc_32bit'
    print(opath16)
    print(opath32)
    # exit()


    # load data
    input_files = get_file_list()
    idx_RGB = get_index()

    # for idx, f in enumerate(range(8)):
    #     arr = get_image()
    for idx, f in enumerate(input_files):
        img = Image.open(f)
        arr = np.asarray(img) # [0, 255]

        print('arr:  %d, %d' %(np.amin(arr), np.amax(arr)))
        patternized = (arr * idx_RGB).astype(np.float32)  # [0, 255]

        arr_input = (patternized / 127.5) - 1  # [0, 255] --> [0, 2] --> [-1, 1]
        arr_input = arr_input[None, ...]

        print('arr_input: %.2f, %.2f' %(np.amin(arr_input), np.amax(arr_input)))




        print(arr.shape, idx_RGB.shape, patternized.shape, patternized.dtype)

        name = os.path.join(opath16, "in%04d.png" % (idx))
        io.imsave(name, arr)
        name = os.path.join(opath16, "pat%04d.png" % (idx))
        io.imsave(name, patternized)

        name = os.path.join(opath32, "in%04d.png" % (idx))
        io.imsave(name, arr)
        name = os.path.join(opath32, "pat%04d.png" % (idx))
        io.imsave(name, patternized)

        # continue


        #inference
        output16 = model16.predict(arr_input)
        output32 = model32.predict(arr_input)
        # output16 = model16(patternized)
        # output32 = model32(patternized)

        print(arr.shape, idx_RGB.shape, patternized.shape, output16.shape, output32.shape)

        # output16 = np.clip((output16 * 2 - 1), 0, 1)
        # output32 = np.clip((output32 * 2 - 1), 0, 1)

        # output16 = np.clip((output16+1)/2, 0, 1)
        # output32 = np.clip((output32+1)/2, 0, 1)

        output16 = np.clip((output16 + 1) * 127.5, 0, 255).astype(np.uint8)
        output32 = np.clip((output32 + 1) * 127.5, 0, 255).astype(np.uint8)

        results16 = np.squeeze(output16)
        results32 = np.squeeze(output32)


        print(np.amin(output16), np.amax(output16))
        print(np.amin(output32), np.amax(output32))



        name = os.path.join(opath16, "%04d.png" % (idx))
        io.imsave(name, results16)

        name = os.path.join(opath32, "%04d.png" % (idx))
        io.imsave(name, results32)

        print(opath16)
        print(opath32)

        if idx>9:
            return

    print('done done')

def main(args):
    tfnames = freeze(args)
    inference_tflite(args, tfnames)
    # inference_h5(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_path',
        type=str,
        # default='/dataset/MIT/tfrecords_sub10',
        default='weights',
        help='add noise on dec input')

    parser.add_argument(
        '--model_name',
        type=str,
        default='MC_rmsc',
        help='MC_rmsc')

    parser.add_argument(
        '--model_sig',
        type=str,
        default='_16bit',
        help='model postfix')

    parser.add_argument(
        '--test',
        type=bool,
        default=False,
        help='test bool, True/False')


    args = parser.parse_args()
    main(args)
