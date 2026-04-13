import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision import transforms
from matplotlib import pyplot as plt
import math
from torchvision.utils import save_image
from torch.optim.lr_scheduler import CosineAnnealingLR

if torch.cuda.is_available():
    device = 'cuda'  
elif torch.mps.is_available():
    device='mps'
else:
    device='cpu'

IMAGE_SHAPE = (3, 64, 64)
IMAGE_DIMENSIONS = (1, 1, 1)
COND_SHAPE = (2,) 

load_checkpoint = False
training=False

cond_one_hot = torch.eye(COND_SHAPE[0], device=device)

transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(64),
    transforms.ToTensor()
])

if training:
    # configure dataset path via environment variable or default to local directory
    dataset_path = os.environ.get("CELEBA_PATH", "./data/celeba")
    training_set = CelebA(root=dataset_path, transform=transform, download=False, split='train')
    training_loader = DataLoader(training_set, batch_size=128, shuffle=True, num_workers=16)


class NoiseSchedule:
    def __init__(self, L, s=0.008, device=device):
        self.L = L
        t = torch.linspace(0.0, L, L+1, device=device) / L
        a = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
        a = a / a[0]
        self.beta = (1 - a[1:] / a[:-1]).clip(0.0, 0.99)
        self.alpha = torch.cumprod(1.0 - self.beta, dim=0)
        self.one_minus_beta = 1 - self.beta
        self.one_minus_alpha = 1 - self.alpha
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.sqrt_beta = torch.sqrt(self.beta)
        self.sqrt_1_alpha = torch.sqrt(self.one_minus_alpha)
        self.sqrt_1_beta = torch.sqrt(self.one_minus_beta)

    def __len__(self):
        return self.L

L = 1000
noise_schedule = NoiseSchedule(L)

print('sqrt(alpha_L)=', noise_schedule.sqrt_alpha[-1].cpu().item(), flush=True)

TIME_ENCODING_SIZE = 64

class TimeEncoding:
    def __init__(self, L, dim, device=device):

        self.L = L
        self.dim = dim
        dim2 = dim // 2
        encoding = torch.zeros(L, dim)
        ang = torch.linspace(0.0, torch.pi/2, L)
        logmul = torch.linspace(0.0, math.log(40), dim2)
        mul = torch.exp(logmul)
        for i in range(dim2):
            a = ang * mul[i]
            encoding[:, 2*i] = torch.sin(a)
            encoding[:, 2*i+1] = torch.cos(a)
        self.encoding = encoding.to(device=device)

    def __len__(self):
        return self.L

    def __getitem__(self, t):
        return self.encoding[t]

time_encoding = TimeEncoding(L, TIME_ENCODING_SIZE)

class UNetBlock(nn.Module):
    def __init__(self, size, outer_features, inner_features, cond_features, inner_block=None):
        super().__init__()
        self.size = size
        self.outer_features = outer_features
        self.inner_features = inner_features
        self.cond_features = cond_features
        
        self.encoder = self.build_encoder(outer_features + 3 * cond_features, inner_features)
        self.decoder = self.build_decoder(inner_features + 3 * cond_features + TIME_ENCODING_SIZE, outer_features)
        self.combiner = self.build_combiner(2 * outer_features, outer_features)
        self.inner = inner_block

    def forward(self, x, time_encodings, male, no_beard, glasses):
        x0 = x
        
        male_cond = male.view(-1, self.cond_features, 1, 1).expand(-1, -1, self.size, self.size)
        no_beard_cond = no_beard.view(-1, self.cond_features, 1, 1).expand(-1, -1, self.size, self.size)
        glasses_cond = glasses.view(-1, self.cond_features, 1, 1).expand(-1, -1, self.size, self.size)
        
        
        x = torch.cat((x, male_cond, no_beard_cond, glasses_cond), dim=1)
        y = self.encoder(x)
        
        if self.inner:
            y = self.inner(y, time_encodings, male, no_beard, glasses)
            
        half_size = self.size // 2
        
        male_cond = male.view(-1, self.cond_features, 1, 1).expand(-1, -1, half_size, half_size)
        no_beard_cond = no_beard.view(-1, self.cond_features, 1, 1).expand(-1, -1, half_size, half_size)
        glasses_cond = glasses.view(-1, self.cond_features, 1, 1).expand(-1, -1, half_size, half_size)
        tt = time_encodings.view(-1, TIME_ENCODING_SIZE, 1, 1).expand(-1, -1, half_size, half_size)
        
        y1 = torch.cat((y, male_cond, no_beard_cond, glasses_cond, tt), dim=1)
        x1 = self.decoder(y1)
        x2 = torch.cat((x1, x0), dim=1)
        return self.combiner(x2)

    def build_combiner(self, from_features, to_features):
        return nn.Conv2d(from_features, to_features, 1)

    def build_encoder(self, from_features, to_features):
        model = nn.Sequential(
            nn.Conv2d(from_features, from_features, 3, padding='same', bias=False),
            nn.BatchNorm2d(from_features),
            nn.ReLU(),
            nn.Conv2d(from_features, to_features, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(to_features),
            nn.ReLU()
        )
        return model

    def build_decoder(self, from_features, to_features):
        model = nn.Sequential(
            nn.Conv2d(from_features, from_features, 3, padding='same', bias=False),
            nn.BatchNorm2d(from_features),
            nn.ReLU(),
            nn.ConvTranspose2d(from_features, to_features, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(to_features),
            nn.ReLU()
        )
        return model

class DDPMUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding='same'),  
            nn.ReLU())
        self.unet = self.build_unet(64, [64, 128, 256, 512, 1024])  
        self.post = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding='same')) 

    def forward(self, x, t, male, no_beard, glasses):
        enc = time_encoding[t]
        x = self.pre(x)
        y = self.unet(x, enc, male, no_beard, glasses)
        y = self.post(y)
        return y

    def build_unet(self, size, feat_list):
        if len(feat_list) > 2:
            inner_block = self.build_unet(size // 2, feat_list[1:])
        else:
            inner_block = None
        return UNetBlock(size, feat_list[0], feat_list[1], COND_SHAPE[0], inner_block)

model = DDPMUNet()
model = model.to(device=device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)

if load_checkpoint:
    checkpoint = torch.load("checkpoints/diff_last2_intermediate/checkpoint310.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

if training:
    print("started training", flush=True)

epoch_count = 0

def training_epoch(dataloader):
    global epoch_count
    model.train()
    average_loss = 0.0
    for x, target in dataloader:
        x = x.to(device=device)
        n = x.shape[0]
        
        
        male = cond_one_hot[target[:, 20]].float()
        no_beard = cond_one_hot[target[:, 24]].float()
        glasses = cond_one_hot[target[:, 15]].float()

        
        P = 0.2
        u = torch.rand((n,))
        male[u < P, :] = 0.0
        no_beard[u < P, :] = 0.0
        glasses[u < P, :] = 0.0

        
        t = torch.randint(0, L, (n,), device=device)
        
        eps = torch.randn_like(x)

        sqrt_alpha = noise_schedule.sqrt_alpha[t].view(-1, *IMAGE_DIMENSIONS)
        sqrt_1_alpha = noise_schedule.sqrt_1_alpha[t].view(-1, *IMAGE_DIMENSIONS)
        zt = sqrt_alpha * x + sqrt_1_alpha * eps

        g = model(zt, t, male, no_beard, glasses)
        loss = loss_function(g, eps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        average_loss = 0.9 * average_loss + 0.1 * loss.cpu().item()
    
    epoch_count += 1
    scheduler.step()
    print(f'Epoch {epoch_count} completed. Average loss={average_loss}', flush=True)

if training:
    for i in range(500):
        training_epoch(training_loader)
        if i % 10 == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict' : scheduler.state_dict(),
                    'epoch': i,
                }, f"checkpoints/diffusion_last2_intermediate/checkpoint{i}.pth")
    

def generate(model, male, no_beard, glasses, lam):
    
    model.eval()
    n = male.shape[0]
    z = torch.randn(n, *IMAGE_SHAPE, device=device)

    male0 = torch.zeros_like(male)
    no_beard0 = torch.zeros_like(no_beard)
    glasses0 = torch.zeros_like(glasses)

    for kt in reversed(range(L)):
        t = torch.full((n,), kt, device=device, dtype=torch.long)

        beta = noise_schedule.beta[t].view(-1, 1, 1, 1)
        sqrt_1_alpha = noise_schedule.sqrt_1_alpha[t].view(-1, 1, 1, 1)
        sqrt_1_beta = noise_schedule.sqrt_1_beta[t].view(-1, 1, 1, 1)
        sqrt_beta = noise_schedule.sqrt_beta[t].view(-1, 1, 1, 1)

        g1 = model(z, t, male, no_beard, glasses)
        g0 = model(z, t, male0, no_beard0, glasses0)
        g = lam * g1 + (1 - lam) * g0

        mu = (z - beta / sqrt_1_alpha * g) / sqrt_1_beta

        if kt > 0:
            eps = torch.randn_like(z)
            z = mu + sqrt_beta * eps
        else:
            z = mu
    return z


if training:
    print("training completed", flush=True)
