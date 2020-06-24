from PIL import Image,ImageDraw
import random
from random import randrange

class Cutout:
    """
    Input: a numpy array

    Args:
    p = probability of applying the augmentation

    spotlight_radius: the length of square

    spot_counts: how many squares to form if p = 1

    img_dim: dimension of image on which it will be applied

    Output: a numpy array

    """


    def __init__(self, p, spotlight_radius=10, spot_counts = 100, img_dim = 256):
        self.p = p
        self.spotlight_radius = spotlight_radius
        self.spot_counts = spot_counts
        self.img_dim = img_dim

    def get_draw(self, img):
        draw = ImageDraw.Draw((img))
        return draw


    def __call__(self, img):
        img = Image.fromarray(img)
        img = img.convert('RGB')   # just to ensure that the image has only three channels
        set_points = [(randrange(self.img_dim), randrange(self.img_dim)) for _ in range(self.spot_counts)]
        draw = self.get_draw(img)
        
        if random.random() < self.p:
            for (x, y) in set_points:
                draw.rectangle((x-self.spotlight_radius, y-self.spotlight_radius, x+self.spotlight_radius, y+self.spotlight_radius), fill = "black")
            img = np.array(img)
            return img
        else:
            img = np.array(img)
            return img
