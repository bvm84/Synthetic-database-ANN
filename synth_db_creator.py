import numpy as np
from pathlib import PurePath
import os
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import NullLocator
import glob


def create_square_array(amplitude, square_duration, full_duration, offset):
    array = np.zeros(shape=1)
    for i in range(full_duration):
        if i < offset:
            array = np.append(array, [0], axis=0)
        elif offset <= i <= offset + square_duration:
            array = np.append(array, [amplitude], axis=0)
        else:
            array = np.append(array, [0], axis=0)
    # print(array)
    return array


def create_square_monodb(amplitude, square_duration, full_duration, quantity, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    n = int(full_duration / quantity)
    for i in range(quantity):
        offset = n * i
        array = create_square_array(amplitude, square_duration, full_duration, offset)
        # print(array)
        filename = folder.joinpath(str(i)).with_suffix('.npy')
        np.save(str(filename), array)


def create_square_db(amp_max, amp_min, step, square_duration, full_duration, quantity, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    n = int(full_duration / quantity)
    for j in range(amp_min, amp_max, step):
        for i in range(quantity):
            offset = n * i
            array = create_square_array(j, square_duration, full_duration, offset)
            # print(array)
            filename = folder.joinpath((str(j) + '_' + str(i))).with_suffix('.npy')
            np.save(str(filename), array)


def create_triangle_array(amplitude, triangle_duration, full_duration, offset):
    koef = int(2 * amplitude / (triangle_duration))
    array = np.zeros(shape=1)
    j = 0
    for i in range(full_duration):
        if i < offset:
            array = np.append(array, [0], axis=0)
        elif offset <= i <= offset + triangle_duration / 2:
            if j <= int(triangle_duration / 2):
                value = koef * j
                # print(value)
                array = np.append(array, [value], axis=0)
                j = j + 1
        elif offset + triangle_duration / 2 <= i <= offset + triangle_duration:
            j = j - 1
            value = koef * (j)
            array = np.append(array, [value], axis=0)
        else:
            array = np.append(array, [0], axis=0)
    # print(array)
    return array


def create_triangle_monodb(amplitude, triangle_duration, full_duration, quantity, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    n = int(full_duration / quantity)
    for i in range(quantity):
        offset = n * i
        array = create_triangle_array(amplitude, triangle_duration, full_duration, offset)
        # print(array)
        filename = folder.joinpath(str(i)).with_suffix('.npy')
        np.save(str(filename), array)


def create_triangle_db(amp_max, amp_min, step, triangle_duration, full_duration, quantity, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for j in range(amp_min, amp_max, step):
        n = int(full_duration / quantity)
        for i in range(quantity):
            offset = n * i
            array = create_triangle_array(j, triangle_duration, full_duration, offset)
            # print(array)
            filename = folder.joinpath((str(j) + '_' + str(i))).with_suffix('.npy')
            np.save(str(filename), array)


def convert_series_to_recurence_matrix(array):
        # print(len(trunc_signal_fd))
        phase_space_dots = np.zeros(shape=[0, 2])
        for i in range(len(array) - 1):
            phase_space_dots = np.append(phase_space_dots, [[array[i], array[i + 1]]], axis=0)
        dims = (len(phase_space_dots), len(phase_space_dots))
        recurence_matrix = np.zeros(dims)
        for i in range(len(phase_space_dots)):
            for j in range(len(phase_space_dots)):
                # calculate Euclidean distance
                recurence_matrix[i, j] = int(math.sqrt(pow((phase_space_dots[i][0] - phase_space_dots[j][0]), 2) +
                                                       pow((phase_space_dots[i][1] - phase_space_dots[j][1]), 2)))
        # plot_recurence(recurence_matrix)
        return recurence_matrix


def save_recurence_image(recurence_matrix, image_name):
    w = 3.93
    h = 3.93
    # this sizes produce picture 400x400 pixels
    DPI = 100
    fig_ws = plt.figure(figsize=(w, h), dpi=DPI, frameon=False)
    ax = fig_ws.add_axes([0, 0, 1, 1])
    plt.imshow(recurence_matrix, cmap='gray', vmin=0, vmax=255)
    # plt.axis('off')
    # ax.set_xlim([0, len(data)])
    # ax.set_ylim([0, 1])
    ax.set_axis_off()
    ax.margins(0, 0.01)
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(NullLocator())
    plt.savefig(image_name, bbox_inches='tight', pad_inches=0, format='png')
    plt.cla()
    plt.clf()
    plt.close('all')


def create_image_db(source_folder, out_folder):
    if not os.path.exists(source_folder):
        print('Source folder does not exist')
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for item in glob.iglob(str(source_folder) + '/*.npy'):
        out_filename = item.split('\\')[-1].split('.')[0]
        out_full_filename = out_folder.joinpath(out_filename).with_suffix('.png')
        array = np.load(item)
        recurence_matrix = convert_series_to_recurence_matrix(array)
        save_recurence_image(recurence_matrix, out_full_filename)


def show_series(filename):
    array = np.load(str(filename))
    print(array)
    plt.plot(array)
    plt.show()


def plot_recurence(recurence_matrix):
    plt.imshow(recurence_matrix, cmap='gray')
    plt.show()


if __name__ == "__main__":
    square_db_folder = PurePath(os.getcwd(), 'square_db')
    triangle_db_folder = PurePath(os.getcwd(), 'triangle_db')
    square_im_db_folder = PurePath(os.getcwd(), 'im_square_db')
    triangle_im_db_folder = PurePath(os.getcwd(), 'im_triangle_db')
    create_square_db(amp_min=1, amp_max=255, step=1,
                     square_duration=100, full_duration=1002, quantity=100, folder=square_db_folder)
    create_triangle_db(amp_min=1, amp_max=255, step=1,
                       triangle_duration=100, full_duration=1002, quantity=100, folder=triangle_db_folder)
    # create_image_db(square_db_folder, square_im_db_folder)
    # create_image_db(triangle_db_folder, triangle_im_db_folder)
    # i = 95
    # filename_to_show = PurePath(os.getcwd(), 'square_db')
    # filename_to_show = PurePath(os.getcwd(), 'triangle_db')
    # show_series(filename_to_show.joinpath(str(i)).with_suffix('.npy'))
