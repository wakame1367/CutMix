import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.backend import image_data_format


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
    bb_x_1 = np.clip(r_x - r_w // 2, 0, width)
    bb_y_1 = np.clip(r_y - r_h // 2, 0, height)
    bb_x_2 = np.clip(r_x + r_w // 2, 0, width)
    bb_y_2 = np.clip(r_y + r_h // 2, 0, height)
    return bb_x_1, bb_y_1, bb_x_2, bb_y_2


def main():
    np.random.seed(2019)
    (x_train, y_train), (_, _) = mnist.load_data()
    label = 1
    target_label = 2
    labels = y_train[y_train == label]
    target_labels = y_train[y_train == target_label]
    print(labels.shape)
    print(target_labels.shape)
    l_rand_idx = np.random.randint(labels.shape[0])
    t_rand_idx = np.random.randint(target_labels.shape[0])
    image = np.transpose(x_train[None, l_rand_idx], (1, 2, 0))
    target_image = np.transpose(x_train[None, t_rand_idx], (1, 2, 0))
    plt.subplot(221)
    plt.imshow(x_train[l_rand_idx], cmap=plt.get_cmap('gray'))
    plt.subplot(222)
    plt.imshow(x_train[t_rand_idx], cmap=plt.get_cmap('gray'))

    print(image.shape)
    print(target_image.shape)
    rand_l = np.random.beta(0.2, 0.2)
    print(rand_l)
    bx1, by1, bx2, by2 = get_rand_bbox(image, rand_l)
    print("{}, {}, {}, {}".format(bx1, by1, bx2, by2))
    target_image[bx1:bx2, by1:by2, :] = image[bx1:bx2, by1:by2, :]
    plt.subplot(223)
    plt.imshow(image[bx1:bx2, by1:by2, :].transpose((2, 0, 1))[0],
               cmap=plt.get_cmap('gray'))
    plt.subplot(224)
    plt.imshow(target_image.transpose((2, 0, 1))[0], cmap=plt.get_cmap('gray'))
    plt.savefig("images/image.png")
    plt.close()


if __name__ == '__main__':
    main()
