import numpy as np
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.backend import image_data_format
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.utils import to_categorical


def is_channel_last(image):
    channel = image.shape[2]
    assert len(image.shape) == 3
    assert channel == 3 or channel == 1
    assert image_data_format() == "channels_last"


def get_rand_bbox(image, l):
    # Note image is channel last
    is_channel_last(image)
    width = image.shape[0]
    height = image.shape[1]
    r_x = np.random.randint(width)
    r_y = np.random.randint(height)
    r_l = np.sqrt(1 - l)
    r_w = np.int(width * r_l)
    r_h = np.int(height * r_l)
    bb_x_1 = np.int(np.clip(r_x - r_w, 0, width))
    bb_y_1 = np.int(np.clip(r_y - r_h, 0, height))
    bb_x_2 = np.int(np.clip(r_x + r_w, 0, width))
    bb_y_2 = np.int(np.clip(r_y + r_h, 0, height))
    return bb_x_1, bb_y_1, bb_x_2, bb_y_2


def main():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = ResNet50(input_shape=(32, 32, 3), include_top=False, weights='imagenet', classes=10)
    model.summary()


if __name__ == '__main__':
    main()
