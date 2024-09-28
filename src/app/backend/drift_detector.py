import os
import numpy as np
from torch.utils.data import DataLoader, _utils, RandomSampler
from pytorch_lightning import LightningDataModule
from src.models.data_loader import MROasis3Datamodule
import torchdrift, torch
from torch.utils.data import DataLoader, RandomSampler, _utils

dir = os.path.dirname(__file__)
model_dir = os.path.join(dir, "..", "..", "..", "models")

class CorruptedDataModule(LightningDataModule):
    def __init__(self, parent: MROasis3Datamodule, additional_transform: callable):
        self.train_dataset = parent.train_dataset
        self.val_dataset = parent.val_dataset
        self.test_dataset = parent.test_dataset
        self.train_batch_size = parent.train_batch_size
        self.val_batch_size = parent.val_batch_size
        self.additional_transform = additional_transform

        self.prepare_data()
        self.setup('fit')
        self.setup('test')

    def setup(self, typ):
        pass

    def collate_fn(self, batch):
        batch = _utils.collate.default_collate(batch)
        batch = (self.additional_transform(batch[0]), *batch[1:])
        return batch

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=4, shuffle=True, collate_fn=self.collate_fn)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, num_workers=4, shuffle=False, collate_fn=self.collate_fn)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, num_workers=4, shuffle=False, collate_fn=self.collate_fn)

    def default_dataloader(self, batch_size=None, num_samples=None, shuffle=True):
        dataset = self.val_dataset
        if batch_size is None:
            batch_size = self.val_batch_size
        replacement = num_samples is not None
        if shuffle:
            sampler = RandomSampler(dataset, replacement=replacement, num_samples=num_samples)
        else:
            sampler = None
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=self.collate_fn)

def gaussian_noise(image: np.ndarray, corruption_level = 1):
    """ Gaussian-distributed additive noise. """
    row, col, ch= image.shape
    mean = 0
    var = 0.1 * corruption_level
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy

def sp_noise(image: np.ndarray, corruption_level = 1):
    """ Adds salt & pepper noise to the image """
    # row,col,ch = image.shape
    s_vs_p = 0.5
    amount = 0.004 * corruption_level
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in image.shape]
    out[coords] = 0
    return out

def poisson_noise(image: np.ndarray, corruption_level = 1):
    """ Poisson-distributed noise generated from the data."""
    vals = len(np.unique(image)) * corruption_level
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy

def speckle_noise(image: np.ndarray, corruption_level = 1):
    """ Multiplicative noise using out = image + n*image,where
        n is uniform noise with specified mean & variance """
    row,col,ch = image.shape
    gauss = np.random.randn(row,col,ch) * corruption_level
    gauss = gauss.reshape(row,col,ch)        
    noisy = image + image * gauss
    return noisy

def corruption_function(image: np.ndarray, mode: str = None) -> np.ndarray:
    """ Function applied to the dataset to measure the drifting of the data """

    noises = ['gauss', 'poisson', 's&p', 'speckle']

    if not mode:
        noises_n = np.random.randint(1, 5)
        for i in noises_n:
            random_noise_index = np.random.randint(0, len(noises))
            mode = noises[random_noise_index]

            if mode == "gauss":
                image = gaussian_noise(image, i)
            elif mode == "s&p":
                image = sp_noise(image, i)
            elif mode == "poisson":
                image = poisson_noise(image, i)
            elif mode =="speckle":
                image = speckle_noise(image, i)
    return image

def main():

    original_datamodule = datamodule # ref to original data module
    corrupted_datamodule = CorruptedDataModule(parent=original_datamodule, additional_transform=corruption_function)

    # Here a batch of original images and the corrupted couterpart should be printed out

    inputs, _ = next(iter(corrupted_datamodule.default_dataloader(shuffle=True)))
    inputs_ood = corruption_function(inputs)

    # N = 6
    # model.eval()
    # inps = torch.cat([inputs[:N], inputs_ood[:N]])
    # model.cpu()
    # predictions = model.predict(inps).max(1).indices

    # predicted_labels = [["ant","bee"][p] for p in predictions]
    # pyplot.figure(figsize=(15, 5))
    # for i in range(2 * N):
    #     pyplot.subplot(2, N, i + 1)
    #     pyplot.title(predicted_labels[i])
    #     pyplot.imshow(inps[i].permute(1, 2, 0))
    #     pyplot.xticks([])
    #     pyplot.yticks([])
    
    # features_path = os.path.join(model_dir, "model.ckpt")
    # feature_extractor # still don't get this

    drift_detector = torchdrift.detectors.KSDriftDetector()
    torchdrift.utils.fit(original_datamodule.train_dataloader(), feature_extractor, drift_detector)


    drift_detection_model = torch.nn.Sequential(
        feature_extractor,
        drift_detector
    )

    features = feature_extractor(inputs)
    score = drift_detector(features)
    p_val = drift_detector.compute_p_value(features)
    score, p_val

    # N_base = drift_detector.base_outputs.size(0)
    # mapper = sklearn.manifold.Isomap(n_components=2)
    # base_embedded = mapper.fit_transform(drift_detector.base_outputs)
    # features_embedded = mapper.transform(features)
    # pyplot.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
    # pyplot.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
    # pyplot.title(f'score {score:.2f} p-value {p_val:.2f}')

    # features = feature_extractor(inputs_ood)
    # score = drift_detector(features)
    # p_val = drift_detector.compute_p_value(features)

    # features_embedded = mapper.transform(features)
    # pyplot.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
    # pyplot.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
    # pyplot.title(f'score {score:.2f} p-value {p_val:.2f}');