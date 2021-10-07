import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Disc128, Gen128, init_weights
from tqdm import tqdm
from utils import *
import os
import shutil

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyper-parameters
TOTAL_EPOCHS = 200

BATCH_MULTIPLIER = 1.8
BATCH_SIZE = int(64 * BATCH_MULTIPLIER)

# Image parameters
IMG_CHANNELS = 3

# Optimizer parameters
LR = 1e-4 * (BATCH_MULTIPLIER ** 0.5)
BETAS = (0.0, 0.9)

# Gen parameters
GEN_FEATURES = 32
Z_SIZE = 100

# Disc parameters
DISC_FEATURES = 32
DISC_ITERATIONS = 5
LAMBDA_GP = 10

# Model saving parameters
LOAD_MODEL = False

MODEL_FP = "FCC_GAN.model"
TRAINING_INFO_FP = "training.info"

# Dataset stuff
DATASET_PATH = "F:/ML_Datasets/amineDataset256"

transforms = transforms.Compose(
    [
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)])
    ]
)

dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Tensorboard stuffs
STEP_SIZE = 40
LOG_DIR_PATH = "logs"

if LOAD_MODEL:
    print("Loading model and training info...")

    # Loads models and training info
    gen, disc = load_models(MODEL_FP)

    gen.train()
    disc.train()

    step, fixedNoise = load_training_info(TRAINING_INFO_FP)

    step += 1
else:
    # Removes tensorboard logs for fresh start
    if os.path.exists(LOG_DIR_PATH):
        shutil.rmtree(LOG_DIR_PATH)

    # Init gen and disc
    gen = Gen128(IMG_CHANNELS, Z_SIZE, GEN_FEATURES).to(device)
    disc = Disc128(IMG_CHANNELS, DISC_FEATURES).to(device)

    init_weights(gen)
    init_weights(disc)

    fixedNoise = torch.randn((32, Z_SIZE)).to(device)

    step = 0

realWriter = SummaryWriter(f"{LOG_DIR_PATH}/real")
fakeWriter = SummaryWriter(f"{LOG_DIR_PATH}/fake")

# Initialize optimizers
genOptim = optim.Adam(gen.parameters(), lr=LR, betas=BETAS)
discOptim = optim.Adam(disc.parameters(), lr=LR, betas=BETAS)

gen.train()
disc.train()

# Training loop
for epoch in range(TOTAL_EPOCHS):
    with tqdm(total=len(loader), desc=f"Epoch {epoch}/{TOTAL_EPOCHS}") as bar:
        for batchIdx, (real, _) in enumerate(loader):
            real = real.to(device)
            batchSize = real.shape[0]

            # Train disc
            for _ in range(DISC_ITERATIONS):
                noise = torch.randn((batchSize, Z_SIZE)).to(device)
                fake = gen(noise)

                discReal = disc(real)
                discFake = disc(fake)

                gp = gradient_penalty(disc, real, fake, device=device)
                discLoss = -(torch.mean(discReal) - torch.mean(discFake)) + LAMBDA_GP * gp

                disc.zero_grad()
                discLoss.backward(retain_graph=True)
                discOptim.step()

            # Train gen
            out = disc(fake)
            genLoss = -torch.mean(out)

            gen.zero_grad()
            genLoss.backward()
            genOptim.step()

            # Shows progress, saves models, and training info every STEP_SIZE of batches
            if batchIdx % STEP_SIZE == 0 and batchIdx != 0:
                gen.eval()
                disc.eval()

                with torch.no_grad():
                    fake = gen(fixedNoise)

                    realImgGrid = torchvision.utils.make_grid(real[:32], normalize=True)
                    fakeImgGrid = torchvision.utils.make_grid(fake[:32], normalize=True)

                    realWriter.add_image("Real", realImgGrid, global_step=step)
                    fakeWriter.add_image("Fake", fakeImgGrid, global_step=step)

                gen.train()
                disc.train()

                save_models(gen, disc, MODEL_FP)
                save_training_info(step, fixedNoise, TRAINING_INFO_FP)

                step += 1

            bar.update(1)
            bar.desc = f"Epoch {epoch}/{TOTAL_EPOCHS} Loss D: {discLoss:.4f}, Loss G: {genLoss:.4f}"
