import cv2 as cv
import numpy as np
from skimage import io, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma

img = img_as_float(cv.imread('IMG_8073.jpg'))
# img = cv.imread('C:/Users/Peter/OneDrive/Pictures/PythonImgs/IMG_2355.jpg')
sigma_est = np.mean(estimate_sigma(img, multichannel=True))
denoise_img = denoise_nl_means(img, h=2.15*sigma_est, fast_mode=True,
                               patch_size=36,
                               patch_distance=8,
                               multichannel=True)

cv.imwrite('NL.jpg', denoise_img)
cv.namedWindow('Original', 0)
cv.resizeWindow('Original', 600, 800)
cv.imshow('Original', img)
cv.namedWindow('Denoised', 0)
cv.resizeWindow('Denoised', 600, 800)
cv.imshow('Denoised', denoise_img)
cv.waitKey(0)
cv.destroyAllWindows()
