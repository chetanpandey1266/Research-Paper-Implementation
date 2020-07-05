import torch 
import torch.nn as nn



class Unet(nn.Module):

    def double_conv(self, in_c, out_c):
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size = 3),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_c, out_c, kernel_size= 3),
            nn.ReLU(inplace = True)
        )
        return conv

    def crop_img(self, tensor, target_tensor):
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta//2
        return tensor[:,:,delta:tensor-delta, delta:tensor_size-delta]

    def __init__(self): 
        super(Unet, self).__init__()
        self.conv1 = self.double_conv(1, 64)
        self.conv2 = self.double_conv(64, 128)
        self.conv3 = self.double_conv(128, 256)
        self.conv4 = self.double_conv(256, 512)
        self.conv5 = self.double_conv(512, 1024)
        self.conv6 = self.double_conv(512, 1024)
        self.conv7 = self.double_conv(512, 1024)
        self.conv8 = self.double_conv(512, 1024)
        self.conv9 = self.double_conv(512, 1024)
        self.conv10 = self.double_conv(512, 1024)

        self.max_pool = nn.MaxPool2d(2, stride = 2)

        self.conv11 = self.double_conv(1024, 512)
        self.conv12 = self.double_conv(512, 256)
        self.conv13 = self.double_conv(256, 128)
        self.conv14 = self.double_conv(128, 64)

    def forward(self, x):
        x1 = self.conv1(x)#
        x2 = self.max_pool(x1)
        x3 = self.conv2(x2)#
        x4 = self.max_pool(x3)
        x5 = self.conv3(x4)#
        x6 = self.max_pool(x5)
        x7 = self.conv4(x6)#
        x8 = self.max_pool(x7)
        x9 = self.conv5(x8)#
        x10 = nn.ConvTranspose2d(1024, 512, 2, 2)(x9)
        x10 = torch.cat([x7, x10])
        x11 = self.conv11(x10)
        x12 = nn.ConvTranspose2d(512, 256, 2, 2)(x11)
        x12 = torch.cat([x5, x12])
        x13 = self.conv12(x12)
        x14 = nn.ConvTranspose2d(256, 128, 2, 2)(x13)
        x14 = torch.cat9([x3, x14])
        x15 = self.conv13(x14)
        x16 = nn.ConvTranspose2d(128, 64, 2, 2)(x15)
        x16 = torch.cat([x1, x16])
        x17 = self.conv14(x16)
        x18 = nn.Conv2d(64, 2, kernel_size = 1)(x17)
        #print(x18.size())



if __name__ == "__main__":
    img = torch.rand(1, 1, 572, 572)
    unet = Unet()
    unet(img)



