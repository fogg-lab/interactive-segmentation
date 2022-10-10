from pathlib import Path
import numpy as np
import cv2
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class TubesDataset(ISDataset):
    def __init__(self, dataset_path, split, custom_augmentor=None, **kwargs):
        super(TubesDataset, self).__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        self._data_path = self.dataset_path / split
        self._images_path = self._data_path / 'images'
        self._insts_path = self._data_path / 'masks'
        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.png'))]
        self._custom_augmentor = custom_augmentor

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        mask_name = image_name.replace("image", "mask")
        image_path = str(self._images_path / image_name)
        mask_path = str(self._insts_path / mask_name)

        image = cv2.imread(image_path, 0)
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        instances_mask = cv2.imread(mask_path, 0)
        instances_mask[instances_mask > 0] = 1

        if image is None or len(image.shape) != 3:
            print(f"Failed to read {image_path}")
        if instances_mask is None or len(instances_mask.shape) != 2:
            print(f"Failed to read {mask_path}")

        return DSample(image, instances_mask, object_ids=[1], sample_id=index,
                       custom_augmentor=self._custom_augmentor)
