import numpy as np
from scipy import ndimage

def apply_depthwise_convolution(image, kernel):
    output_image = np.zeros_like(image)
    for k in range(image.shape[2]):
        output_image[:, :, k] = ndimage.convolve(image[:, :, k], kernel[:, :, k], mode='constant', cval=0.0)
    return output_image

def apply_pointwise_convolution(image, kernel):
    # This assumes a simple channel-wise multiplication followed by sum, mimicking a depthwise operation
    # It does not correspond to the classical definition of pointwise convolution in CNNs
    output_image = np.zeros(image.shape[:-1])  # Prepare output image without channel dimension
    for k in range(image.shape[2]):
        output_image += image[:, :, k] * kernel[0, 0, k]  # Simple multiplication with kernel and sum
    return output_image

def execute_convolution():
    input_image = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                            [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
                            [[2, 4, 6], [8, 1, 3], [5, 7, 9]]])
    kernel = np.array([[[1, 0, -1], [1, 0, -1], [1, 0, -1]],
                       [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
                       [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]])

    convolution_type = input("Enter 'depthwise' or 'pointwise' for convolution: ")

    if convolution_type == 'depthwise':
        result_image = apply_depthwise_convolution(input_image, kernel)
    elif convolution_type == 'pointwise':
        result_image = apply_pointwise_convolution(input_image, kernel)
    else:
        print("Invalid convolution type. Please enter 'depthwise' or 'pointwise'.")
        return

    print("Convoluted Image:")
    print(result_image)

if __name__ == "__main__":
    execute_convolution()
