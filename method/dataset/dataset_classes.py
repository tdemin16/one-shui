import csv
import multiprocessing as mp
import numpy as np
import os
import random
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from typing import Optional, List


class IdentityUnlearningDataset:
    def __init__(
        self,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        num_identities: int,
        splits: Optional[dict] = None,
    ):
        """
        Parameters:
        -----
        train_dataset: torch.utils.data.Dataset
            Dataset with training augmentations
        test_dataset: torch.utils.data.Dataset
            Dataset with test augmentations
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        # number of identities to forget
        self.num_identities = num_identities
        self.num_all_identities = len(self.train_dataset.get_identity_to_images())

        self.TRAIN = []
        self.VAL = []
        self.TEST = []

        self.support_set = []
        self.FORGET = []
        self.RETAIN = []

        self.simulated_support_set = []

        if splits is None:
            self._split_data()
            self.RETAIN, self.FORGET, self.support_set = self._init_unlearning(self.TRAIN)
        else:
            # splits are provided when retraining or unlearning to avoid wrong splits
            self.TRAIN = splits["train"]
            self.VAL = splits["val"]
            self.TEST = splits["test"]
            self.support_set = splits["support"] if "support" in splits else splits["support_set"]
            self.FORGET = splits["forget"]
            self.RETAIN = splits["retain"]

        # consistency checks, let's make sure nothing overlaps
        assert set(self.TRAIN) & set(self.TEST) == set()
        assert set(self.TRAIN) & set(self.VAL) == set()
        assert set(self.TEST) & set(self.VAL) == set()
        assert set(self.FORGET) & set(self.RETAIN) == set()
        assert set(self.FORGET) | set(self.RETAIN) == set(self.TRAIN)
        
        # check that support set images do not overlap with training ones
        assert set(self.support_set) & set(
            [
                img
                for img, id_ in enumerate(self.train_dataset.get_identities())
                if id_ in self.TRAIN
            ]
        ) == set(self.support_set)

    def get_splits(self):
        return {
            "train": self.TRAIN,
            "val": self.VAL,
            "test": self.TEST,
            "support": self.support_set,
            "forget": self.FORGET,
            "retain": self.RETAIN,
        }

    def get_num_classes(self):
        return self.train_dataset.get_num_classes()

    def get_data(self, identities, train):
        """
        Get images from :identities: without the support set.
        """
        # use train or test augmentations
        dataset = self.train_dataset if train else self.test_dataset
        samples = []
        # use sets for faster lookup
        set_identities = set(identities)
        set_support_set = set(self.support_set)
        # itherate thorugh image, identity pairs
        for img_id, identity in enumerate(dataset.get_identities()):
            if (
                identity in set_identities  # image in the identity set
                and not img_id in set_support_set  # image not in the support set
            ):
                samples.append(img_id)
        return torch.utils.data.Subset(dataset, samples)

    def get_train_data(self, train=True):
        """
        Retrieve all training data without the support set

        Args:
            train (bool, optional): Whether to use train or test augmentations. Defaults to True.
        """
        return self.get_data(self.TRAIN, train)

    def get_simulated_unlearning_data(self, num_ids=None):
        """
        Returns a simulated forget request
        """

        set_support_set = set(self.support_set)
        identity2images = self.test_dataset.get_identity_to_images()

        # remove ids that are in the test and validation sets
        if isinstance(identity2images, dict):
            identity2images = {
                identity: images
                for identity, images in identity2images.items()
                if identity in self.TRAIN
            }
        else:
            identity2images = {
                identity: images
                for identity, images in enumerate(identity2images)
                if identity in self.TRAIN
            }

        # get all training identities
        all_train_identities = set(self.TRAIN)

        # shuffle identities
        import copy

        identities = copy.deepcopy(self.TRAIN)
        random.shuffle(identities)

        datasets = []
        self.simulated_support_set = {}

        num_identities = self.num_identities if num_ids is None else num_ids

        # num identities in each unlearning request. extra identities are added to fill the batch
        for i in range(0, len(identities), num_identities):
            support_set_split = []
            forget_set_split = []
            retain_set_split = []

            if i + num_identities < len(identities):
                ids = identities[i : i + num_identities]
            else:
                ids = identities[i:] + identities[: i + num_identities - len(identities)]

            # remove ids that are in forget set
            retain_ids = all_train_identities - set(ids)

            for identity in ids:
                # retrieve forget images and remove support set if present
                set_images = set()
                set_images.update(identity2images[identity])
                set_images -= set_support_set
                valid_images = list(set_images)

                # extract one simulated support set and the rest as forgetting set
                image = random.sample(valid_images, 1)[0]
                support_set_split.append(image)
                self.simulated_support_set[identity] = image

                # add the rest of the images to the forget set
                if len(valid_images) > 1:
                    valid_images.remove(image)
                forget_set_split.extend(valid_images)

            for identity in retain_ids:
                retain_images = identity2images[identity]
                retain_images = list(set(retain_images) - set_support_set)
                retain_set_split.extend(retain_images)

            datasets.append(
                [
                    {
                        "support": torch.utils.data.Subset(self.test_dataset, support_set_split),
                        "forget": torch.utils.data.Subset(self.train_dataset, forget_set_split),
                        "retain": torch.utils.data.Subset(self.train_dataset, retain_set_split),
                    },
                    {
                        "support": torch.utils.data.Subset(self.test_dataset, support_set_split),
                        "forget": torch.utils.data.Subset(self.test_dataset, forget_set_split),
                        "retain": torch.utils.data.Subset(self.test_dataset, retain_set_split),
                    },
                ]
            )

        return datasets

    def get_val_data(self, train: bool = False) -> torch.utils.data.Subset:
        """
        Retrieve all validation data from the validation set
        """
        return self.get_data(self.VAL, train=train)

    def get_test_data(self):
        """
        Retrieve all test data from the test set
        """
        return self.get_data(self.TEST, train=False)

    def get_unlearning_data(self, train=False):
        """
        Retrieve all unlearning data
        """
        datasets = {
            "support": torch.utils.data.Subset(self.test_dataset, self.support_set),
            "forget": self.get_data(self.FORGET, train),
            "retain": self.get_data(self.RETAIN, train),
        }
        return datasets

    def _split_using_target_ratio(self, ratio, identities):
        # target number of images
        set_identities = set(identities)
        images = [
            img
            for img, identity in enumerate(self.train_dataset.get_identities())
            if identity in set_identities
        ]
        target = len(images) * ratio

        # get first guess on splitting point as target % of the identities
        limit = int(len(identities) * ratio)
        test_ids = set(identities[:limit])
        num_test_images = len(
            [
                img
                for img, identity in enumerate(self.train_dataset.get_identities())
                if identity in test_ids
            ]
        )

        # if the number of test images is greater than the target, move the limit to the left
        last_side = -1 if num_test_images > target else +1

        # move the limit until the number of test images is close to the target
        while (num_test_images > target and last_side == -1) or (
            num_test_images <= target and last_side == +1
        ):
            limit += last_side
            num_test_images += last_side * len(
                self.train_dataset.get_identity_to_images()[identities[limit - 1]]
            )

        print(
            f"Num images: {len(images)}({len(identities)}) - Ratio: {ratio} - Num test images: {num_test_images} - Target: {target}"
        )
        return identities[:limit], identities[limit:]

    def _split_data(self):
        """
        Split the dataset into training and testing sets based on the identities while keeping
        a ratio train/test of 20%.
        """

        # create a shuffled list of identities
        if isinstance(self.train_dataset.get_identity_to_images(), dict):
            shuffled_identities = list(self.train_dataset.get_identity_to_images().keys())
        else:
            shuffled_identities = list(range(len(self.train_dataset.get_identity_to_images())))
        random.shuffle(shuffled_identities)

        intermediate, self.TRAIN = self._split_using_target_ratio(
            ratio=0.4,
            identities=shuffled_identities,
        )
        self.VAL, self.TEST = self._split_using_target_ratio(
            ratio=0.5,
            identities=intermediate,
        )

    def _init_unlearning(self, train_ids):
        """
        Finds :num_identities: identities with more than 2 images and
        selects one image from each identity as the support sample.
        The rest of the images are used as retain set.
        """
        num_identities_so_far = 0

        # shuffling
        identity2images = self.train_dataset.get_identity_to_images()
        if isinstance(identity2images, dict):
            identities = list(identity2images.keys())
        else:
            identities = list(range(len(identity2images)))
        random.shuffle(identities)

        RETAIN = []
        FORGET = []
        support_set = []

        # iterate through shuffled identities
        for identity in identities:
            images = identity2images[identity]
            if identity in train_ids:
                # add image to forget set if identity has more than 1 images
                # and quota for identities is not reached
                # and if num_identities is 1 then we want at least 6 images otherwise we cannot compute acc
                if (
                    len(images) > 1
                    and num_identities_so_far < self.num_identities
                    and ((self.num_identities == 1 and len(images) > 5) or self.num_identities > 1)
                ):
                    image = random.sample(images, 1)[0]
                    support_set.append(image)
                    FORGET.append(identity)
                    num_identities_so_far += 1
                else:
                    RETAIN.append(identity)

        return RETAIN, FORGET, support_set


class IdentityDataset(torch.utils.data.Dataset):
    def get_num_classes(self):
        raise NotImplementedError

    def get_identities(self):
        raise NotImplementedError

    def get_identity_to_images(self):
        raise NotImplementedError


class MUFAC(IdentityDataset):
    def __init__(self, root, transform=None):
        super().__init__()

        self.root = root
        self.transform = transform

        self.fpath = os.path.join(self.root, "mufac")

        self.data = []
        self.identities = []
        self.ages = []
        self.age_map = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
        self.identity_to_images = []
        self.classes = [
            "0-6 years old",
            "7-12 years old",
            "13-19 years old",
            "20-30 years old",
            "31-45 years old",
            "46-55 years old",
            "56-66 years old",
            "67-80 years old",
        ]

        self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        identity = self.identities[index]
        age = self.ages[index]
        return img, identity, age

    def get_num_classes(self):
        return len(self.classes)

    def get_identities(self):
        return self.identities

    def get_identity_to_images(self):
        return self.identity_to_images

    def _load_data(self):
        # since the dataset annotation does not provide the id number we must compute it
        seen_ids = {}  # maps seen string ids into integer ids
        curr_id_num = 0  # gets increased every time we find a new id

        # we do the train val test split by ourselves, let's collect all data
        for split in ("train", "val", "test"):
            # get annotation file for the split
            annotation_file = f"custom_{split}_dataset.csv"
            annotation_path = os.path.join(self.fpath, annotation_file)

            with open(annotation_path, "r") as fp:
                reader = csv.reader(fp)
                for i, row in enumerate(reader):
                    # skip firs row
                    if i > 0:
                        # join the first two entry in the row
                        id_string = "_".join(row[:2])
                        if id_string not in seen_ids:
                            seen_ids[id_string] = curr_id_num
                            curr_id_num += 1

                        self.data.append(os.path.join(self.fpath, f"{split}_images", row[-1]))
                        self.identities.append(seen_ids[id_string])
                        self.ages.append(self.age_map[row[2]])
        
        self.identity_to_images = [[] for _ in range(max(self.identities) + 1)]
        for i, identity in enumerate(self.identities):
            self.identity_to_images[identity].append(i)


class CelebA(IdentityDataset, torchvision.datasets.CelebA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.identity_to_images = {}
        for i, identity in enumerate(self.get_identities()):
            if identity not in self.identity_to_images:
                self.identity_to_images[identity] = []
            self.identity_to_images[identity].append(i)

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return image, target[1], target[0]

    def get_num_classes(self):
        return self.attr.size(1)

    def get_identities(self):
        return self.identity.squeeze().tolist()

    def get_identity_to_images(self):
        return self.identity_to_images


class CelebAHQ(IdentityDataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform

        self.fpath = os.path.join(self.root, "CelebAMask-HQ")
        self.image_folder = os.path.join(self.fpath, "CelebA-HQ-img")
        self.identity_file = os.path.join(self.fpath, "CelebA-HQ-identity.txt")
        self.attribute_file = os.path.join(self.fpath, "CelebAMask-HQ-attribute-anno.txt")

        self.data = []  # img_id -> img_path
        self.identities = []  # img_id -> person_id
        self.attributes = []  # img_id -> [attr_i, attr_j, attr_k, ...]
        self.identity_to_images = []  # person_id -> [img_id_i, img_id_j, img_id_k, ...]
        self.classes = None

        self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        identity = self.identities[index]
        attributes = torch.nn.functional.one_hot(
            torch.LongTensor(self.attributes[index]), num_classes=len(self.classes)
        ).sum(dim=0)
        return img, identity, attributes

    def get_num_classes(self):
        return len(self.classes)

    def get_identities(self):
        return self.identities

    def get_identity_to_images(self):
        return self.identity_to_images

    def _load_data(self):
        with open(self.attribute_file, "r") as f:
            # first line: num_images
            self.num_images = int(f.readline().strip())
            self.data = [None] * self.num_images
            self.attributes = [None] * self.num_images

            # second line: class names
            self.classes = f.readline().strip().split()
            for line in f:
                # entry: img_id.jpg attr1 attr2 ... attr40
                entry = line.strip().split()
                img_id = int(entry[0].split(".")[0])

                img_path = os.path.join(self.image_folder, entry[0])
                self.data[img_id] = img_path
                self.attributes[img_id] = [i for i, attr in enumerate(entry[1:]) if int(attr) != -1]

        self.identities = [-1] * self.num_images
        with open(self.identity_file, "r") as f:
            for line in f:
                # entry: img_id.jpg person_id
                entry = line.strip().split()
                img_id = int(entry[0].split(".")[0])
                person_id = int(entry[1])

                self.identities[img_id] = person_id

        self.identity_to_images = [[] for _ in range(max(self.identities) + 1)]
        for i, identity in enumerate(self.identities):
            self.identity_to_images[identity].append(i)
