import imgaug as ia
import imgaug.augmenters as iaa

"""
Example from: https://github.com/aleju/imgaug
"""

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential(
    [
        sometimes(iaa.CropAndPad(
            percent=(-0.03, 0.03),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},  # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},  # translate by -20 to +20 percent (per axis)
            rotate=(-15, 15),  # rotate by -45 to +45 degrees
            shear=(-3, 3),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
                   [
                       # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                       # convert images into their superpixel representation
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 1.0)),  # blur images with a sigma between 0 and 3.0
                           iaa.AverageBlur(k=(1, 1.5)),
                           # blur image using local means with kernel sizes between 2 and 7
                           iaa.MedianBlur(k=(1, 3)),
                           # blur image using local medians with kernel sizes between 2 and 7
                       ]),
                       iaa.Sharpen(alpha=(0, 0.6), lightness=(0.75, 1.2)),  # sharpen images
                       iaa.Emboss(alpha=(0, 1.0), strength=(0, 1.0)),  # emboss images
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                       # add gaussian noise to images
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                           iaa.CoarseDropout((0.03, 0.1), size_percent=(0.02, 0.05), per_channel=0.2),
                       ]),
                       iaa.LinearContrast((0.5, 1.2), per_channel=0.5),  # improve or worsen the contrast
                       # move pixels locally around (with random strengths)
                       sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.02))),  # sometimes move parts of the image around
                       sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                   ],
                   random_order=True
                   )
    ],
    random_order=True
)
