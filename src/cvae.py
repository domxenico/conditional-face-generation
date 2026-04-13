import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
import time
import os

LATENT_SIZE= 128
device="cuda" if torch.cuda.is_available() else 'cpu'
beta=1.0
reconstruction_loss_function=nn.BCELoss(reduction='sum')
BATCH_SIZE=64


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
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
    

class ConditionalVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            ConvolutionalBlock(6, 64, kernel_size=3, stride=1, padding=1),
            ConvolutionalBlock(64, 128),
            ConvolutionalBlock(128, 256),
            ConvolutionalBlock(256, 512),
            ConvolutionalBlock(512, 512, kernel_size=4, stride=2, padding=1),
            nn.Flatten(), 
            nn.Linear(512*4*4, 1024),
            nn.ReLU()
        )

        self.conditional_vector=nn.Sequential(
            nn.Linear(6, 128), 
            nn.ReLU(), 
            nn.Linear(128, 256)
        )

        self.mu=nn.Linear(1024+256, LATENT_SIZE)
        self.log_sigma=nn.Linear(1024+256, LATENT_SIZE)

        self.decoder=nn.Sequential(
            nn.Linear(LATENT_SIZE+6, 256*2*2),
            nn.ReLU(),
            nn.Linear(256*2*2, 512*4*4),
            nn.Unflatten(1, (512, 4, 4)),
            ConvTranspBlock(512, 256),
            ConvTranspBlock(256, 128),
            ConvTranspBlock(128, 64),
            ConvTranspBlock(64, 16),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.eye=torch.eye(2).to(device)


    def forward(self, x, male, no_beard, glasses):
        conditional_tensor= self.create_conditional_tensor(x, male, no_beard, glasses)
        x_cond= torch.cat((x, conditional_tensor), dim=1)
        out_enc=self.encoder(x_cond)
        cond_vector=self.conditional_vector(self.create_conditional_vector(male, no_beard, glasses))
        out_enc=torch.cat([out_enc, cond_vector], dim=1)
        mu=self.mu(out_enc)
        log_sigma=self.log_sigma(out_enc)
        eps= torch.randn_like(log_sigma).to(device)
        z= mu + eps * torch.exp(log_sigma)
        z_cond= torch.cat((z, self.create_conditional_vector(male, no_beard, glasses)), dim=1)
        return self.decoder(z_cond), mu, log_sigma
    
    def generate_one_sample(self, z, male, no_beard, glasses):
        male= self.eye[male].unsqueeze(0)
        no_beard= self.eye[no_beard].unsqueeze(0)
        glasses= self.eye[glasses].unsqueeze(0)
        z_cond= torch.cat((z, male, no_beard, glasses), dim=1)
        return self.decoder(z_cond)
    
    def generate_more_samples(self, z, male, no_beard, glasses):
        z_cond= torch.cat((z, self.create_conditional_vector(male, no_beard, glasses)), dim=1)
        return self.decoder(z_cond)

    def create_conditional_tensor(self, x, male, no_beard, glasses):
        batch_size, _, H, W = x.shape
        conditional_tensor = torch.zeros(batch_size, 3, H, W, device=device, dtype=torch.float32)
        conditional_tensor[:, 0, :, :] = male.view(-1, 1, 1).float().to(device)
        conditional_tensor[:, 1, :, :] = (1 - no_beard).view(-1, 1, 1).float().to(device)  
        conditional_tensor[:, 2, :, :] = glasses.view(-1, 1, 1).float().to(device)

        return conditional_tensor
    
    def create_conditional_vector(self, male, no_beard, glasses):
       return torch.cat([F.one_hot(male, 2).float().to(device), F.one_hot((no_beard), 2).float().to(device),F.one_hot(glasses, 2).float().to(device)], dim=1)
    
        
def training_loop(model, training_loader, optimizer, epochs=10, start_epoch=0):
    start_time=time.time() 
    for i in range(start_epoch, start_epoch+epochs):
        loss_logger=[]
        for x, attr in training_loader:
            x= x.to(device)
            male = attr[:, 20].long()
            no_beard= attr[:, 24].long()
            glasses = attr[:, 15].long()
            generated_image, mu, log_sigma=model(x, male, no_beard, glasses)
            optimizer.zero_grad()
            loss=loss_function(generated_image, x, mu, log_sigma)
            loss_logger.append(loss.item())
            loss.backward()
            optimizer.step()
        avg_loss=sum(loss_logger)/len(loss_logger)
        current_time=time.time()
        if current_time - start_time >= SAVING_INTERVAL:
            start_time = current_time
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': i+1,
                'loss': avg_loss,
            }, f"checkpoints/cvae_prova3_intermediate/checkpoint_{i}.pth")
            
        print(f"Epoch {i+1}, Loss: {avg_loss}", flush=True)

    return model.state_dict(), optimizer.state_dict(), start_epoch + epochs - 1 , avg_loss
        

def kl_loss_function(mu, log_sigma):
    kl=0.5*(mu**2 + torch.exp(2*log_sigma)-1-2*log_sigma)
    return torch.sum(kl)

def loss_function(reconstructed, original, mu, log_sigma):
    return reconstruction_loss_function(reconstructed, original) + \
           beta*kl_loss_function(mu, log_sigma)

if __name__ == "__main__":
    # configure dataset path via environment variable or default to local directory
    path_to_dataset = os.environ.get("CELEBA_PATH", "./data/celeba")
    SAVING_INTERVAL=600

    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64), 
        transforms.ToTensor()
    ])

    training_set=CelebA(path_to_dataset, split="all", transform=transform, download=False, target_type='attr')

    training_loader=DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
    model=ConditionalVAE()
    model.to(device)
    model.train()
    optimizer=optim.AdamW(model.parameters(), lr=0.0001)
    checkpoint_path="final_weights/cvae_terza_prova3.pth"
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"resuming training from epoch {start_epoch} with loss {checkpoint['loss']}", flush=True)
    else: 
        start_epoch=0
    
    model_state, opt_state, curr_epoch, avg_loss=training_loop(model, training_loader, optimizer, epochs=100, start_epoch=start_epoch)

    torch.save({    'model_state_dict': model_state,
                    'optimizer_state_dict': opt_state,
                    'epoch': curr_epoch,
                    'loss': avg_loss,
                }, 'final_weights/cvae_terza_prova4.pth')


