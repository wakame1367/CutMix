import numpy as np
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.backend import image_data_format
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.utils import to_categorical


class CutMixGenerator:
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2,
                 shuffle=True, data_gen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.data_gen = data_gen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        _, class_num = self.y_train.shape
        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        y1 = self.y_train[batch_ids[:self.batch_size]]
        y2 = self.y_train[batch_ids[self.batch_size:]]
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X = X1 * X_l + X2 * (1 - X_l)
        y = y1 * y_l + y2 * (1 - y_l)

        if self.data_gen:
            for i in range(self.batch_size):
                X[i] = self.data_gen.random_transform(X[i])

        return X, y


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
