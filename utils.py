import torch


def save_models(gen, disc, model_fp):
    torch.save({"gen": gen, "disc": disc}, model_fp)


def load_models(model_fp):
    modelDict = torch.load(model_fp)

    return modelDict["gen"], modelDict["disc"]


def save_training_info(step, fixed_noise, training_info_fp):
    torch.save({"step": step, "fixedNoise": fixed_noise}, training_info_fp)


def load_training_info(training_info_fp):
    trainingInfoDict = torch.load(training_info_fp)

    return trainingInfoDict["step"], trainingInfoDict["fixedNoise"]


def gradient_penalty(disc, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)

    interpolatedImages = real * alpha + fake * (1 - alpha)

    mixedScores = disc(interpolatedImages)

    gradient = torch.autograd.grad(
        inputs=interpolatedImages,
        outputs=mixedScores,
        grad_outputs=torch.ones_like(mixedScores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradientNorm = gradient.norm(2, dim=1)
    gradientPenalty = torch.mean((gradientNorm - 1) ** 2)

    return gradientPenalty
