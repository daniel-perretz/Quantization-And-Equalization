import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from imageio import imwrite
from skimage.color import rgb2gray


MAX_PIXEL = 255
GRAYSCALE = 1
RGB = 2
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                          [0.596, -0.275, -0.321],
                                          [0.212, -0.523, 0.311]])


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    im = imread(filename)  # rgb return type
    if (representation == GRAYSCALE and im.ndim == 3):
        gray_scale_img = rgb2gray(im)
        return np.float64(gray_scale_img)
    else:

        return np.float64(im / MAX_PIXEL)


def imdisplay(filename, representation):
    """
    Reads an image and displays it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    """
    img = read_image(filename, representation)
    if (representation == GRAYSCALE):
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space
    :param imRGB: height X width X 3 np.float64 matrix in the [0,1] range
    :return: the image in the YIQ space
    """
    return np.dot(imRGB, RGB_YIQ_TRANSFORMATION_MATRIX.T)


def yiq2rgb(imYIQ):
    """
    Transform a YIQ image into the RGB color space
    :param imYIQ: height X width X 3 np.float64 matrix in the [0,1] range for
        the Y channel and in the range of [-1,1] for the I,Q channels
    :return: the image in the RGB space
    """
    inverse = \
        np.linalg.inv(RGB_YIQ_TRANSFORMATION_MATRIX)

    Y = imYIQ[:, :, 0]
    I = imYIQ[:, :, 1]
    Q = imYIQ[:, :, 2]

    R = inverse[0:1, 0:1] * Y + inverse[0:1, 1:2] * I + inverse[0:1, 2:3] * Q
    G = inverse[1:2, 0:1] * Y + inverse[1:2, 1:2] * I + inverse[1:2, 2:3] * Q
    B = inverse[2:3, 0:1] * Y + inverse[2:3, 1:2] * I + inverse[2:3, 2:3] * Q
    rgb = np.dstack((R, G, B))
    return rgb


def histogram_equalize(im_orig):
    """
    Perform histogram equalization on the given image
    :param im_orig: Input float64 [0,1] image
    :return: [im_eq, hist_orig, hist_eq]
    """
    orig = im_orig
    if(im_orig.ndim == 3):
        yiq_img = rgb2yiq(im_orig)
        im_orig = yiq_img[:,:,0]

    im_orig = (im_orig*255).astype(int)
    hist_orig, bins = np.histogram(im_orig, bins=256, range=(0,255))
    cum_hist = np.cumsum(hist_orig)


    pixel_num = np.max(cum_hist)
    first_none_arr = np.ma.masked_equal(cum_hist,0)
    first_none_zero = first_none_arr.min()

    T_k = (cum_hist-first_none_zero)/ (pixel_num-first_none_zero)*MAX_PIXEL
    T_k = np.ceil(T_k)
    im_orig = im_orig.astype((np.int64))
    new_img = T_k[im_orig]
    new_img_hist, bins = np.histogram(new_img, bins=256)
    new_img = new_img / 255

    if (np.ndim(orig) == 3):
        yiq_img[:,:,0] = new_img
        orig = yiq2rgb(yiq_img)
        return [orig,hist_orig,new_img_hist]

    return [new_img,hist_orig,new_img_hist]



def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input float64 [0,1] image
    :param n_quant: Number of intensities im_quant image will have
    :param n_iter: Maximum number of iterations of the optimization
    :return:  im_quant - is the quantized output image
              error - is an array with shape (n_iter,) (or less) of
                the total intensities error for each iteration of the
                quantization procedure
    """

    orig = im_orig
    if(im_orig.ndim == 3):
        yiq_img = rgb2yiq(im_orig)
        im_orig = yiq_img[:,:,0]

    new_map = np.empty(256).astype(np.int64)
    im_orig = im_orig*255
    im_orig = im_orig.astype(np.int64)
    hist_orig,bins = np.histogram(im_orig,bins=256)
    cum_hist = np.cumsum(hist_orig)
    num_of_pixels_each_segment = cum_hist.max()/(n_quant)
    z_array = np.array([])
    for i in range(1,n_quant):
        z_array = np.append(z_array,np.where(cum_hist >=
                                             i*num_of_pixels_each_segment)[0][0])

    z_array = np.append(z_array,[MAX_PIXEL,0])
    z_array.sort()
    z_array = z_array.astype(np.int64)# bounds
    q_array = np.empty(n_quant)
    range_vec = np.arange(z_array[0],z_array[1]+1)
    hist_vec = hist_orig[z_array[0]:z_array[1]+1]
    temp = sum(hist_orig[z_array[0]:z_array[1]+1])
    q_array[0] =  np.dot(range_vec,hist_vec.T)//temp

    z_array_orig = z_array.copy() # so that I can check the changings
    error_array = np.array([])
    grey_level_arr = np.arange(256)
    for i in range(n_iter):
        for j in range(1,n_quant):
            range_vec = np.arange(z_array[j]+1,z_array[j+1]+1)
            hist_vec = hist_orig[z_array[j]+1:z_array[j+1]+1]
            temp = sum(hist_orig[z_array[j]+1:z_array[j+1]+1])
            q_array[j] = np.dot(range_vec,hist_vec.T)//temp

        for p in range(1,n_quant):
            z_array[p] = (q_array[p-1] + q_array[p])//2


        #check converge
        if((z_array ==z_array_orig).all()):
            break
        else:
            z_array_orig = z_array
        #calculate the error
        for ind in range(n_quant):
            sum_err = 0
            sum_err += (((q_array[ind] - grey_level_arr
            [z_array[ind]+1:z_array[ind+1]+1])**2)*
                        hist_orig[z_array[ind]+1:z_array[ind+1]+1]).sum()
            sum_err = sum_err.astype(int)

        error_array = np.append(error_array,sum_err)



        for ind in range(n_quant):
            new_map[z_array[ind]:z_array[ind+1]] = q_array[ind]
        new_map[z_array[-2]:z_array[-1]+1] = q_array[-1]


    new_img = new_map[im_orig]/255
    if (np.ndim(orig) == 3):
        yiq_img[:,:,0] = new_img
        new_img = yiq2rgb(yiq_img)
        return [new_img,error_array]

    return [new_img,error_array.astype(int)]
