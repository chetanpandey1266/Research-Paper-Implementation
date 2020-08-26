import torch 
import torchvision
import torch.nn as nn 
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator

# Hyper Parameters
lr = 0.0002
batch_size = 64
image_size = 64 # We are using MNIST which has a size 28x28 which is to be resized to 64X64

channels_img = 1
channels_noise = 256

num_epochs = 10

features_d = 16
features_g = 16

my_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
])

dataset = datasets.MNIST(root='dataset/', train=True, transform=my_transforms, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

netD = Discriminator(channels_img, features_d).to(device)
netG = Generator(channels_noise, channels_img, features_g).to(device)

optimD = optim.Adam(netD.parameters(), lr = lr)
optimG = optim.Adam(netG.parameters(), lr = lr)

netG.train()
netD.train()

criterion = nn.BCELoss()

real_label = 1
fake_label = 0

input_noise = torch.randn(64, channels_noise, 1, 1).to(device)

#writer_fake = SummaryWriter(f'runs/GAN_MNIST/test_fake')
#writer_real = SummaryWriter(f'runs/GAN_MNIST/test_real')

print("Training..............")

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.to(device)
        targets = targets.to(device)
        batch_size = data.shape[0]
        ## Train Dicriminator: max log(D(x)) + log(1-D(G(z)))
        netD.zero_grad()
        label = (torch.ones(batch_size)*0.9).to(device)
        # we multiply the ones with some fraction close to 1 so that the model do not become very confident on its predictions
        output = netD(data).reshape(-1)
        lossD_real = criterion(output, label)
        D_x = output.mean().item()
        
        noise = torch.randn(batch_size, channels_noise, 1, 1).to(device)
        fake = netG(noise).to(device)
        label = (torch.ones(batch_size)*0.1).to(device)
        
        output = netD(fake.detach()).reshape(-1)  # we do detach to stop its backpropagation through netG
        
        lossD_fake = criterion(output, label)

        lossD = lossD_fake + lossD_real
        lossD.backward()
        optimD.step()


        ## Train Generator: max log(D(G(z)))
        netG.zero_grad()
        label = torch.ones(batch_size).to(device)
        output = netD(fake).reshape(-1)
        lossG = criterion(output, label)
        lossG.backward()
        optimG.step()



        if(batch_idx%100 == 0):
            print(f'Epoch [{epoch}/{num_epochs}] Batch[{batch_idx}/{len(dataloader)}] Loss D: {lossD:.4f}, LossG: {lossG:.4f} D(x): {D_x:.4f}')

            with torch.no_grad():
                fake = netG(input_noise)
                
                img_grid_real = torchvision.utils.make_grid(data[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                #writer_real.add_image('Mnits Real Images', img_grid_real)
                #writer_real.add_image('Mnist Fake Images', img_grid_fake)


