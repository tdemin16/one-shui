############################################################################
# Zero-Shot Machine Unlearning at Scale via Lipschitz Regularization
# Code adapted from https://github.com/jwf40/Zeroshot-Unlearning-At-Scale/tree/main
############################################################################

import torch

from copy import deepcopy
from torchvision.transforms import v2


# https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0, device="cpu"):
        self.std = std
        self.mean = mean
        self.device = device

    def __call__(self, tensor):
        _max = tensor.max()
        _min = tensor.min()
        tensor = (
            tensor + torch.randn(tensor.size()).to(self.device) * self.std + self.mean
        )
        tensor = torch.clamp(tensor, min=_min, max=_max)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class LipschitzRegularization:
    def __init__(
        self,
        model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        parameters=None,
    ):
        self.model = model
        self.device = device

        self.n_samples = parameters["n_samples"]
        self.learning_rate = parameters["learning_rate"]
        self.lipschitz_weighting = parameters["lipschitz_weighting"]

        self.transforms = v2.Compose([
            AddGaussianNoise(
                0.0, self.lipschitz_weighting, device=self.device
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def modify_weight(
        self,
        forget_dl: torch.utils.data.DataLoader,
    ) -> None:
        """
        Spectral forgetting but first perturb weights based on the SSD equations given in the paper
        Parameters:
         - forget_dl (DataLoader): DataLoader containing the samples to be forgotten
        Returns:
         - None
        """

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # Calculate smoothness penalty

        for x, _, _ in forget_dl:
            x = x.to(self.device)
            image = x.unsqueeze(0) if x.dim() == 3 else x
            out = self.model(image)
            if isinstance(out, tuple):
                out = out[0]
            loss = torch.tensor(0.0, device=self.device)
            out_n = torch.tensor(0.0, device=self.device)
            in_n = torch.tensor(0.0, device=self.device)
            # Build comparison images

            for _ in range(self.n_samples):
                img2 = self.transforms(deepcopy(x))
                image2 = img2.unsqueeze(0) if img2.dim() == 3 else img2

                with torch.no_grad():
                    out2 = self.model(image2)
                    if isinstance(out2, tuple):
                        out2 = out2[0]
                # ignore batch dimension
                flatimg, flatimg2 = image.view(image.size()[0], -1), image2.view(
                    image2.size()[0], -1
                )

                in_norm = torch.linalg.vector_norm(flatimg - flatimg2, dim=1)
                out_norm = torch.linalg.vector_norm(out - out2, dim=1)
                # these are used to print
                in_n += in_norm.sum()
                out_n += out_norm.sum()
                K = ((out_norm / in_norm).sum()).abs()
                loss += K

            # Normalize
            loss /= self.n_samples
            in_n /= self.n_samples
            out_n /= self.n_samples

            loss.backward()
            optimizer.step()
