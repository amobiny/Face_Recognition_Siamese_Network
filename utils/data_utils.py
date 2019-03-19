import re
import numpy as np


def read_image(filename, byteorder='>'):
    # first we read the image, as a raw file to the buffer
    with open(filename, 'rb') as f:
        buffer = f.read()

    # using regex, we extract the header, width, height and maxval of the image
    header, width, height, maxval = re.search(b"(^P5\s(?:\s*#.*[\r\n])*"
                                              b"(\d+)\s(?:\s*#.*[\r\n])*"
                                              b"(\d+)\s(?:\s*#.*[\r\n])*"
                                              b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()

    # then we convert the image to numpy array using np.frombuffer which interprets buffer as one dimensional array
    return np.frombuffer(buffer,
                         dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                         count=int(width) * int(height),
                         offset=len(header)
                         ).reshape((int(height), int(width)))


def get_data(total_sample_size):
    # read the image
    image = read_image('data/s' + str(1) + '/' + str(1) + '.pgm', 'rw+')
    # reduce the size
    size=2
    image = image[::size, ::size]
    # get the new size
    dim1 = image.shape[0]
    dim2 = image.shape[1]

    count = 0

    # initialize the numpy array with the shape of [total_sample, no_of_pairs, dim1, dim2]
    x_geuine_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])  # 2 is for pairs
    y_genuine = np.zeros([total_sample_size, 1])

    for i in range(40):
        for j in range(int(total_sample_size / 40)):
            ind1 = 0
            ind2 = 0

            # read images from same directory (genuine pair)
            while ind1 == ind2:
                ind1 = np.random.randint(10)
                ind2 = np.random.randint(10)

            # read the two images
            img1 = read_image('data/s' + str(i + 1) + '/' + str(ind1 + 1) + '.pgm', 'rw+')
            img2 = read_image('data/s' + str(i + 1) + '/' + str(ind2 + 1) + '.pgm', 'rw+')

            # reduce the size
            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]

            # store the images to the initialized numpy array
            x_geuine_pair[count, 0, 0, :, :] = img1
            x_geuine_pair[count, 1, 0, :, :] = img2

            # as we are drawing images from the same directory we assign label as 1. (genuine pair)
            y_genuine[count] = 1
            count += 1

    count = 0
    x_imposite_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])
    y_imposite = np.zeros([total_sample_size, 1])

    for i in range(int(total_sample_size / 10)):
        for j in range(10):

            # read images from different directory (imposite pair)
            while True:
                ind1 = np.random.randint(40)
                ind2 = np.random.randint(40)
                if ind1 != ind2:
                    break

            img1 = read_image('data/s' + str(ind1 + 1) + '/' + str(j + 1) + '.pgm', 'rw+')
            img2 = read_image('data/s' + str(ind2 + 1) + '/' + str(j + 1) + '.pgm', 'rw+')

            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]

            x_imposite_pair[count, 0, 0, :, :] = img1
            x_imposite_pair[count, 1, 0, :, :] = img2
            # as we are drawing images from the different directory we assign label as 0. (imposite pair)
            y_imposite[count] = 0
            count += 1

    # now, concatenate, genuine pairs and imposite pair to get the whole data
    X = np.concatenate([x_geuine_pair, x_imposite_pair], axis=0) / 255
    Y = np.concatenate([y_genuine, y_imposite], axis=0)

    return X, Y


def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :, :, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def get_next_batch(x, y, start, end):
    x1_batch = np.transpose(x[start:end, 0], [0, 2, 3, 1])
    x2_batch = np.transpose(x[start:end, 1], [0, 2, 3, 1])
    y_batch = y[start:end]
    return x1_batch, x2_batch, y_batch
