from PIL import Image,ImageDraw
import time


class Sprikler(object):
    """
    with the aspect of no distortion and looking from the same view
    """

    reso_width = 0
    reso_height = 0
    radius = 10
    def __init__(self,width,height,spotlight_radius= 10):
        self.reso_width = width
        self.reso_height = height
        self.radius = spotlight_radius

    def get_image_spotlight(self,set_points): #function for drawing spot light
        image,draw = self.get_image()
        for (x,y) in set_points:
            draw.ellipse((x-self.radius,y-self.radius,x+self.radius,y+self.radius),fill = "black")
        image.show("titel")
        return image

    def get_image(self):   #function for drawing black image
        image = Image.open("./flowers.jpeg")#(ImageHandler.reso_width,ImageHandler.reso_height),"black")
        draw = ImageDraw.Draw((image))
        return image,draw



from random import randrange 




if __name__ == "__main__":
    hi = Sprikler(240, 240)
    spot_count= 15
    points = [(randrange(240), randrange(240)) for _ in range(spot_count)]
    img = hi.get_image_spotlight(points)