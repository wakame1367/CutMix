import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.backend import image_data_format


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
    bb_y_1 = np.int(np.clip(r_y - r_h, 0, width))
    bb_x_2 = np.int(np.clip(r_x + r_w, 0, width))
    bb_y_2 = np.int(np.clip(r_y + r_h, 0, width))
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
    plt.imshow(image[bx1:bx2, by1:by2, :].transpose((2, 0, 1))[0], cmap=plt.get_cmap('gray'))
    plt.subplot(224)
    plt.imshow(target_image.transpose((2, 0, 1))[0], cmap=plt.get_cmap('gray'))
    plt.savefig("images/image.png")
    plt.close()


if __name__ == '__main__':
    main()
