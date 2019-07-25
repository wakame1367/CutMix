import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.datasets import mnist

from cutout import get_rand_bbox


def main():
    np.random.seed(2019)
    (x_train, y_train), (_, _) = mnist.load_data()
    alpha = 0.2
    l_rand_idx = np.random.randint(x_train.shape[0])
    t_rand_idx = np.random.randint(x_train.shape[0])
    image = np.transpose(x_train[None, l_rand_idx], (1, 2, 0))
    target_image = np.transpose(x_train[None, t_rand_idx], (1, 2, 0))
    plt.subplot(1, 3, 1)
    plt.imshow(x_train[l_rand_idx], cmap=plt.get_cmap('gray'))
    plt.subplot(1, 3, 2)
    plt.imshow(x_train[t_rand_idx], cmap=plt.get_cmap('gray'))

    print(image.shape)
    print(target_image.shape)
    rand_l = np.random.beta(alpha, alpha)
    print(rand_l)
    bx1, by1, bx2, by2 = get_rand_bbox(image.shape[0], image.shape[1], rand_l)
    print("{}, {}, {}, {}".format(bx1, by1, bx2, by2))
    target_image[bx1:bx2, by1:by2, :] = image[bx1:bx2, by1:by2, :]
    plt.subplot(1, 3, 3)
    plt.imshow(target_image.transpose((2, 0, 1))[0], cmap=plt.get_cmap('gray'))
    plt.savefig("images/mnist.png")
    plt.close()


if __name__ == '__main__':
    main()
