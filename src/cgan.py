import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torch.nn.functional import binary_cross_entropy
import os
from torchvision.utils import save_image


LATENT_SIZE= 128
device="cuda" if torch.cuda.is_available() else 'cpu'
BATCH_SIZE=64
# configure dataset path via environment variable or default to local directory
path_to_dataset = os.environ.get("CELEBA_PATH", "./data/celeba")
SAVING_INTERVAL=3
LABEL_SMOOTHING_CHANGE=7


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.net(x)
    
class ConvTranspBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.net=nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)

class ConditionalGANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator=nn.Sequential(
            nn.Linear(LATENT_SIZE+6, 1024*4*4 ),
            nn.ReLU(),
            nn.Unflatten(1, (1024, 4, 4)),
            ConvTranspBlock(1024, 512),
            ConvTranspBlock(512, 256),
            ConvTranspBlock(256, 128),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        self.eye=torch.eye(2).to(device)

    def forward(self, x, male, no_beard, glasses):
        z=torch.randn(x.shape[0], LATENT_SIZE, device=device)
        z_cond = torch.cat([z, self.create_conditional_vector(male, no_beard, glasses)], dim=1)
        return self.generator(z_cond)

    def create_conditional_vector(self, male, no_beard, glasses):
       return torch.cat([F.one_hot(male, 2).float().to(device), F.one_hot((no_beard), 2).float().to(device),F.one_hot(glasses, 2).float().to(device)], dim=1)
    
    def generate_one_sample(self, z, male, no_beard, glasses):
        male= self.eye[male].unsqueeze(0)
        no_beard= self.eye[no_beard].unsqueeze(0)
        glasses= self.eye[glasses].unsqueeze(0)
        z_cond= torch.cat((z, male, no_beard, glasses), dim=1)
        return self.generator(z_cond)
    
    def generate_more_samples(self, z, male, no_beard, glasses):
        z_cond= torch.cat((z, self.create_conditional_vector(male, no_beard, glasses)), dim=1)
        return self.generator(z_cond)
    
class ConditionalGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolutional_layers=nn.Sequential(
            ConvolutionalBlock(3, 64),
            ConvolutionalBlock(64,128),
            ConvolutionalBlock(128,256),
            ConvolutionalBlock(256,512),
            nn.Flatten(),
        )
        self.conditional_vector=nn.Sequential(
            nn.Linear(6, 128), 
            nn.ReLU(),
            nn.Linear(128, 512)
        )
    
        self.classification_head= nn.Sequential(
            nn.Linear(512*4*4+512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, image, male, no_beard, glasses):
        img_features =self.convolutional_layers(image)
        cond_vector= self.conditional_vector(self.create_conditional_vector(male, no_beard, glasses))
        
        return self.classification_head(torch.cat([img_features, cond_vector], dim=1))


    def create_conditional_vector(self, male, no_beard, glasses):
       return torch.cat([F.one_hot(male, 2).float().to(device), F.one_hot((no_beard), 2).float().to(device),F.one_hot(glasses, 2).float().to(device)], dim=1)


def disc_loss_function(d_true, d_synth, label_smoothing):
    t_true=torch.ones_like(d_true)-label_smoothing
    t_synth=torch.zeros_like(d_synth)+label_smoothing
    return binary_cross_entropy(d_true, t_true) + binary_cross_entropy(d_synth, t_synth)

def gen_loss_function(d_synth):
    t_synth=torch.ones_like(d_synth)
    return binary_cross_entropy(d_synth, t_synth)

def training_loop(generator, discriminator, training_loader, gen_optimizer, disc_optimizer, label_smoothing, epochs=10, start_epoch=0):
    for i in range(start_epoch, start_epoch+epochs):
        gen_loss_logger=[]
        disc_loss_logger=[]
        sum_dtrue=0.0
        sum_dsynth=0.0
        n_batch=0
        for x, attr in training_loader:
            n_batch += 1
            x= x.to(device)
            male = attr[:, 20].long()
            no_beard= attr[:, 24].long()
            glasses = attr[:, 15].long()
            
            synth_images=generator(x, male, no_beard, glasses)
            d_synth=discriminator(synth_images, male, no_beard, glasses)
            d_true=discriminator(x, male, no_beard, glasses)

            disc_optimizer.zero_grad()
            disc_loss=disc_loss_function(d_true, d_synth, label_smoothing)
            disc_loss.backward(retain_graph=True)
            disc_optimizer.step()
            
            d_synth=discriminator(synth_images, male, no_beard, glasses)
            gen_optimizer.zero_grad()
            gen_loss=gen_loss_function(d_synth)
            gen_loss.backward()
            gen_optimizer.step()
            
            gen_loss_logger.append(gen_loss.item())
            disc_loss_logger.append(disc_loss.item())

            sum_dtrue += d_true.detach().mean().cpu().item()
            sum_dsynth += d_synth.detach().mean().cpu().item()

        avg_gen_loss=sum(gen_loss_logger)/len(gen_loss_logger)
        avg_disc_loss=sum(disc_loss_logger)/len(disc_loss_logger)
        avg_d_synth=sum_dsynth/n_batch
        avg_d_true=sum_dtrue/n_batch
            
        if (i+1) % SAVING_INTERVAL == 0:
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'gen_optimizer_state_dict': gen_optimizer.state_dict(),
                'disc_optimizer_state_dict': disc_optimizer.state_dict(),
                'label_smoothing': label_smoothing, 
                'epoch': i,
                'gen_loss': avg_gen_loss,
                'disc_loss': avg_disc_loss,
            }, f'checkpoints/cgan_prova11_intermediate/checkpoint_{i}.pth')

            generator.eval()
            with torch.no_grad():

                epoch_dir = f"checkpoints/cgan_prova11_intermediate/images/epoch_{i+1}"
                os.makedirs(epoch_dir, exist_ok=True)

                z1 = torch.randn(1, LATENT_SIZE).to(device)
                z2 = torch.randn(1, LATENT_SIZE).to(device)
                z3 = torch.randn(1, LATENT_SIZE).to(device)
                z4 = torch.randn(1, LATENT_SIZE).to(device)

                img1 = generator.generate_one_sample(z1, 0, 1, 1)
                img2 = generator.generate_one_sample(z2, 1, 0, 1)
                img3 = generator.generate_one_sample(z3, 0, 1, 0)
                img4 = generator.generate_one_sample(z4, 1, 1, 1)

                save_image(img1, os.path.join(epoch_dir, "image1.png"))
                save_image(img2, os.path.join(epoch_dir, "image2.png"))
                save_image(img3, os.path.join(epoch_dir, "image3.png"))
                save_image(img4, os.path.join(epoch_dir, "image4.png"))
                
            generator.train()

        if (i+1)%LABEL_SMOOTHING_CHANGE ==0 :
            label_smoothing=max(0.05, label_smoothing-0.03)

        if avg_d_true > 0.65 and avg_d_synth < 0.35 :
            for param in disc_optimizer.param_groups:
                param['lr'] *= 0.9
            for param in gen_optimizer.param_groups:
                param['lr'] *= 1.05

        
        print(f"Epoch {i+1}, Generator Loss: {avg_gen_loss}", flush=True)
        print(f"Discriminator Loss: {avg_disc_loss}", flush=True)
        print(f"DTrue= {avg_d_true}", flush=True)
        print(f"DSynth= {avg_d_synth}", flush=True)

    return generator.state_dict(), discriminator.state_dict(), gen_optimizer.state_dict(), disc_optimizer.state_dict(), start_epoch + epochs - 1 , gen_loss, disc_loss, label_smoothing

if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64), 
        transforms.ToTensor()
    ])

    training_set=CelebA(path_to_dataset, split="all", transform=transform, download=False, target_type='attr')

    training_loader=DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
    generator=ConditionalGANGenerator()
    discriminator=ConditionalGANDiscriminator()
    generator.to(device)
    discriminator.to(device)
    gen_optimizer=optim.Adam(generator.parameters(), lr=0.0003, betas = (0.5, 0.999))
    disc_optimizer=optim.Adam(discriminator.parameters(), lr=0.0001, betas = (0.5, 0.999))
    checkpoint_path="checkpoints/cgan_prova11_intermediate/checkpoint_68.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        start_epoch = checkpoint['epoch']
        label_smoothing=checkpoint['label_smoothing']
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        print(f"resuming training from epoch {start_epoch} - gen_loss: {checkpoint['gen_loss']} disc_loss: {checkpoint['disc_loss']}", flush=True)
    else:
        start_epoch=0
        label_smoothing=0.2

    generator.train()
    discriminator.train()
    
    generator_state, discriminator_state, gen_opt_state, disc_opt_state, curr_epoch, gen_loss, disc_loss, label_smoothing= training_loop(generator, discriminator, training_loader, gen_optimizer, disc_optimizer, label_smoothing, epochs=100, start_epoch=start_epoch)

    torch.save({    'generator_state_dict': generator_state,
                    'discriminator_state_dict': discriminator_state,
                    'gen_optimizer_state_dict': gen_opt_state,
                    'disc_optimizer_state_dict': disc_opt_state,
                    'label_smoothing': label_smoothing, 
                    'epoch': curr_epoch,
                    'gen_loss': gen_loss.item(),
                    'disc_loss': disc_loss.item(),
                }, 'final_weights/cgan_undicesima_prova.pth')