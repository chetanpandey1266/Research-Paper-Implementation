import numpy as np
import random
from PIL import Image

class Smart:
    """
    Input:

    p =  probability of applying augmentation

    Args:
        img_list: list of images with same label(dtype: np.float32)
        
        label: label whose image list is passes as first argument(dtype: int)

        weight_list: list of weight applied to the corresponding images to be passed(dtype: float)
        whose sum should be equal to one 

    Output:

        img : resulting image(dtype : np.float32)
        label : as passed in input
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img_list, label, weight_list):
        
        img = 0
        wt_sum = 0
        if self.p > random.random():
            for im, wt in zip(img_list, weight_list):
                img += im*wt
                wt_sum += wt
            img = img/wt_sum
            return img, label
        


if __name__ == "__main__":
    img1 = Image.open(r"img1_path")
    img1 = img1.resize((300, 300))
    img2 = Image.open(r"img2_path")
    img2 = img2.resize((300, 300))
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    print(img1.shape, img2.shape)
    w1, w2 = 0.8, 0.2
    img, label = Smart(p=1)([img1, img2], 1, [w2, w1])
    img =  img.astype(np.uint8)
    image = Image.fromarray(img)
    image.show()
