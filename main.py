import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

from skimage.exposure import match_histograms

source_img_path = 'imgs\\jpegPIA01333.jpg'
reference_img_path = 'imgs\\Lenna.jpg'


def calculate_lookup(src_cdf, ref_cdf):
    """
    This method creates the lookup table
    :param array src_cdf: The cdf for the source image
    :param array ref_cdf: The cdf for the reference image
    :return: lookup_table: The lookup table
    :rtype: array
    """

    lookup_table = np.zeros(256)
    for src_pixel_val in range(len(src_cdf)):
        lookup_val = 0
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table


def print_histogram(_histrogram, name, title):
    plt.figure()
    plt.title(title)
    plt.plot(_histrogram, color='#ef476f')
    plt.bar(np.arange(len(_histrogram)), _histrogram, color='#b7b7a4')
    plt.ylabel('Number of Pixels')
    plt.xlabel('Pixel Value')
    plt.savefig("result\\" + "hist_" + name)


def print_image(_image, name, title):
    plt.figure()
    plt.title(title)
    plt.imshow(_image, cmap='gray')
    plt.savefig("result\\" + "img_" + name)


def match_histograms(src_image, ref_image):
    src_hist, src_bin = np.histogram(src_image.flatten(),
                                     bins=256,
                                     range=[0, 256],
                                     density=True)

    ref_hist, ref_bin = np.histogram(ref_image.flatten(),
                                     bins=256,
                                     range=[0, 256],
                                     density=True)

    print_histogram(src_hist,name="src",title="src image histogram")
    print_histogram(ref_hist, name="ref", title="ref image histogram")

    src_cdf = src_hist.cumsum() * 255
    ref_cdf = ref_hist.cumsum() * 255

    transform = calculate_lookup(src_cdf, ref_cdf)

    match_image = cv2.LUT(src_image, transform)

    match_hist, match_bin = np.histogram(
        match_image.flatten(),
        bins=256,
        range=[0, 256],
        density=True
    )
    match_cdf = match_hist.cumsum() * 255

    plt.figure()
    plt.title("cumulative probability")
    src_plot, = plt.plot(src_cdf, color='r')
    ref_plot, = plt.plot(ref_cdf, color='b')
    match_plot, = plt.plot(match_cdf, color = 'g')
    plt.legend([src_plot, ref_plot, match_plot],["src_cdf", "ref_cdf", 'match_cdf'])
    plt.ylabel('Number of Pixels')
    plt.xlabel('Pixel Value')
    plt.savefig("result\\" + "cumulative probability")

    return match_image, match_hist, transform


def equalize_histogram(image: np.ndarray):
    equ_image = np.zeros_like(image)
    src_hist, src_bin = np.histogram(image.flatten(),
                                     bins=256,
                                     range=[0, 256],
                                     density=True)
    transform = src_hist.cumsum() * 255

    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            equ_image[y, x] = transform[image[y, x]]

    equ_hist, _ = np.histogram(equ_image.flatten(),
                               bins=256,
                               range=[0, 256],
                               density=True)

    return equ_image, equ_hist, transform


def main():
    """
    Main method of the program
    """
    source_gray = cv2.imread(cv2.samples.findFile(source_img_path), cv2.IMREAD_GRAYSCALE)
    reference_gray = cv2.imread(cv2.samples.findFile(reference_img_path), cv2.IMREAD_GRAYSCALE)

    equ_image, equ_hist, src_cdf = equalize_histogram(source_gray)
    print_image(equ_image, "equ", "equalized image")
    print_histogram(equ_hist, "equ", "equalize histogram")
    print_histogram(src_cdf, "cdf", "cdf")

    match_image, match_hist, transform = match_histograms(source_gray, reference_gray)
    print_image(match_image, "matched", "matched image")
    print_histogram(match_hist, "match", "match histogram")
    print_histogram(transform, "match_transfer", "match transfer function")


if __name__ == '__main__':
    main()
    sys.exit(1)
