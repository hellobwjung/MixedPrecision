import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def get_image_from_single_example(example, key='image', num_channels=3, dtype='uint8'):
    patch_size = 128

    feature = {
        key: tf.io.FixedLenFeature((), tf.string)
    }

    parsed = tf.io.parse_single_example(example, feature)

    image = tf.io.decode_raw(parsed[key], out_type=dtype)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [patch_size, patch_size, num_channels])

    return image

def main():
    vizrecord= 'mit_viz_000.tfrecords'
    dataset = tf.data.TFRecordDataset(vizrecord)

    mylist = list(dataset.as_numpy_iterator())

    print(len(mylist), type(mylist[0]))

    for idx, e in enumerate(mylist):
        image = get_image_from_single_example(e)
        print(type(image), type(image.numpy()))

        image = image.numpy()
        print(image.shape, np.amin(image), np.amax(image))


        plt.figure()
        plt.imshow(image.astype(np.uint16))
        plt.show()



        pass





if __name__ == "__main__":
    main()
