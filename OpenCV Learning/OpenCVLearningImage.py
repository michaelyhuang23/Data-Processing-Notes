import cv2
import numpy as np

img1 = cv2.imread('slytherin.jpg')
img2 = cv2.imread('gryffindor.jpg')
img3 = cv2.imread('hallows.jpg')

# clamping addition
add2 = cv2.add(img1, img2)
# add and then clamp any value above 255 to 255 (so relative intensity, aka color is not preserved)
cv2.imshow('clamping add', add2)

# average
avg = cv2.addWeighted(img1, 0.5, img2, 0.5, 50)
# the weighted avg of the two images. The last argument is just a constant added to all channels in all pixels (raising the brightness).
cv2.imshow('average blending', avg)

# now we attempt to impose img3 on avg, we want to get rid of img3's background
img3Gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img3Gray, 220, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('normal thresholding', mask)
# Thresholding applies to all channels independently, so this function is usually used for grayscale image only.
# THRESH_BINARY and its inverse are the type thresholding where pixels lower than the second argument will be converted to zero while higher values will be converted to the maxval argument. Inverse basically inverses that.
# Other thresholding types include:
# cv2.THRESH_TRUNC; it truncates all values larger than the threshold to the threshold value (basically clamping)
# cv2.THRESH_TOZERO; it makes all values smaller than the threshold 0

ret2, mask2 = cv2.threshold(img3Gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# type is of course an integer. we add cv2.THRESH_OTSU so that it encodes both the binary_inv info and the otsu info. The type corresponding to the integer resulting from their addition (8) is the type which is binary_inv and uses otsu algorithm to determine the threshold value.
# The basic idea of OTSU algorithm is to look at the histogram of the pixel value and then determine the threshold value to be the one with least in-class variance.
cv2.imshow('otsu thresholding', mask2)

ret3, mask3 = cv2.threshold(img3Gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
# Triangle is another algorithm for determining the threshold value.

cv2.imshow('triangle thresholding', mask3)

invMask = cv2.bitwise_not(mask)
# a cv2.bitwise operation operates on every bit of every number of every pixel. For normal integer,
# eg: cv2.bitwise_not(0) = -1 (all bits are reversed, including the sign bit).
# For unsigned integer (which is what color channels dtype is), cv2.bitwise_not(0)=255.
# Basically has the effect of 255 - x.
width, height, channels = img3.shape
roi = avg[100:100 + width, 100:100 + height]
# roi : region of image. Note that this is a copied out nparray, it doesn't reference the original nparray

bg = cv2.bitwise_and(roi, roi, mask=invMask)
# the mask argument indicates which pixels to apply to bitwise operation to. If the mask value for a pixel is >0, bitwise operation is applied otherwise, that pixel will be filled with 0. Here we are just using bitwise operation to actually use the masking functionality.
fg = cv2.bitwise_and(img3, img3, mask=mask)
dst = fg + bg
avg[100:100 + width, 100:100 + height] = dst

cv2.imshow('average blending with logo', avg)
# each imshow renders a new image in a new window. If the name is the same, it overwrites
'''
arr = np.array([[1, 2], [34, 54]], dtype='uint8')
# uint8 : unsigned integer of 8 bits
print(arr)
arr2 = cv2.bitwise_not(arr)
print(arr2)
'''

# Adaptive Thresholding, in contrary to global thresholding, applies a different threshold value to a pixel based on the local environment.
img4 = cv2.imread('bookpage.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('original bookpage', img4)
retval1, thres1 = cv2.threshold(img4, 12, 255, cv2.THRESH_BINARY)
cv2.imshow('normal bookpage', thres1)
# it doesn't work well because a global threshold is unsuitable
gaus = cv2.adaptiveThreshold(img4, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY, 101, 1)
# third argument is intuitively the addaptive method. All adaptive method uses a neighborhood area with size indicated by blocksize (the second to last argument) and determines a threshold value using some algorithm than subtract from it a constant C (determined by the last argument)
# Gaussian method computes the gaussian weighted avg in the local region. (gaussian weighted basically mean the avg is weighted by gaussian function. Pixels in the center are weighted exponentially (thus connecting to probability) more than pixels farther away)
# Mean method computes the mean value in the region.
cv2.imshow('gaus bookpage', gaus)

mean = cv2.adaptiveThreshold(img4, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                             cv2.THRESH_BINARY, 21, 1)
cv2.imshow('mean bookpage', mean)

cv2.waitKey(0)
cv2.destroyAllWindows()
