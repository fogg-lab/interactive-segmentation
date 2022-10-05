import cv2
import random
import numpy as np
from PIL import Image
from math import floor
from skimage import measure, morphology

from albumentations.core.serialization import SERIALIZABLE_REGISTRY
from albumentations import ImageOnlyTransform, DualTransform
from albumentations.core.transforms_interface import to_tuple
from albumentations.augmentations import functional as F
from isegm.utils.misc import get_bbox_from_mask, expand_bbox, clamp_bbox, get_labels_with_sizes


class UniformRandomResize(DualTransform):
    def __init__(self, scale_range=(0.9, 1.1), interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.scale_range = scale_range
        self.interpolation = interpolation

    def get_params_dependent_on_targets(self, params):
        scale = random.uniform(*self.scale_range)
        height = int(round(params['image'].shape[0] * scale))
        width = int(round(params['image'].shape[1] * scale))
        return {'new_height': height, 'new_width': width}

    def apply(self, img, new_height=0, new_width=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.resize(img, height=new_height, width=new_width, interpolation=interpolation)

    def apply_to_keypoint(self, keypoint, new_height=0, new_width=0, **params):
        scale_x = new_width / params["cols"]
        scale_y = new_height / params["rows"]
        return F.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args_names(self):
        return "scale_range", "interpolation"

    @property
    def targets_as_params(self):
        return ["image"]


class ZoomIn(DualTransform):
    def __init__(
            self,
            height,
            width,
            bbox_jitter=0.1,
            expansion_ratio=1.4,
            min_crop_size=200,
            min_area=100,
            always_resize=False,
            always_apply=False,
            p=0.5,
    ):
        super(ZoomIn, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.bbox_jitter = to_tuple(bbox_jitter)
        self.expansion_ratio = expansion_ratio
        self.min_crop_size = min_crop_size
        self.min_area = min_area
        self.always_resize = always_resize

    def apply(self, img, selected_object, bbox, **params):
        if selected_object is None:
            if self.always_resize:
                img = F.resize(img, height=self.height, width=self.width)
            return img

        rmin, rmax, cmin, cmax = bbox
        img = img[rmin:rmax + 1, cmin:cmax + 1]
        img = F.resize(img, height=self.height, width=self.width)

        return img

    def apply_to_mask(self, mask, selected_object, bbox, **params):
        if selected_object is None:
            if self.always_resize:
                mask = F.resize(mask, height=self.height, width=self.width,
                                interpolation=cv2.INTER_NEAREST)
            return mask

        rmin, rmax, cmin, cmax = bbox
        mask = mask[rmin:rmax + 1, cmin:cmax + 1]
        if isinstance(selected_object, tuple):
            layer_indx, mask_id = selected_object
            obj_mask = mask[:, :, layer_indx] == mask_id
            new_mask = np.zeros_like(mask)
            new_mask[:, :, layer_indx][obj_mask] = mask_id
        else:
            obj_mask = mask == selected_object
            new_mask = mask.copy()
            new_mask[np.logical_not(obj_mask)] = 0

        new_mask = F.resize(new_mask, height=self.height, width=self.width,
                            interpolation=cv2.INTER_NEAREST)
        return new_mask

    def get_params_dependent_on_targets(self, params):
        instances = params['mask']

        is_mask_layer = len(instances.shape) > 2
        candidates = []
        if is_mask_layer:
            for layer_indx in range(instances.shape[2]):
                labels, areas = get_labels_with_sizes(instances[:, :, layer_indx])
                candidates.extend([(layer_indx, obj_id)
                                   for obj_id, area in zip(labels, areas)
                                   if area > self.min_area])
        else:
            labels, areas = get_labels_with_sizes(instances)
            candidates = [obj_id for obj_id, area in zip(labels, areas)
                          if area > self.min_area]

        selected_object = None
        bbox = None
        if candidates:
            selected_object = random.choice(candidates)
            if is_mask_layer:
                layer_indx, mask_id = selected_object
                obj_mask = instances[:, :, layer_indx] == mask_id
            else:
                obj_mask = instances == selected_object

            bbox = get_bbox_from_mask(obj_mask)

            if isinstance(self.expansion_ratio, tuple):
                expansion_ratio = random.uniform(*self.expansion_ratio)
            else:
                expansion_ratio = self.expansion_ratio

            bbox = expand_bbox(bbox, expansion_ratio, self.min_crop_size)
            bbox = self._jitter_bbox(bbox)
            bbox = clamp_bbox(bbox, 0, obj_mask.shape[0] - 1, 0, obj_mask.shape[1] - 1)

        return {
            'selected_object': selected_object,
            'bbox': bbox
        }

    def _jitter_bbox(self, bbox):
        rmin, rmax, cmin, cmax = bbox
        height = rmax - rmin + 1
        width = cmax - cmin + 1
        rmin = int(rmin + random.uniform(*self.bbox_jitter) * height)
        rmax = int(rmax + random.uniform(*self.bbox_jitter) * height)
        cmin = int(cmin + random.uniform(*self.bbox_jitter) * width)
        cmax = int(cmax + random.uniform(*self.bbox_jitter) * width)

        return rmin, rmax, cmin, cmax

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError

    def apply_to_keypoint(self, keypoint, **params):
        raise NotImplementedError

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_transform_init_args_names(self):
        return ("height", "width", "bbox_jitter",
                "expansion_ratio", "min_crop_size", "min_area", "always_resize")


def remove_image_only_transforms(sdict):
    if not 'transforms' in sdict:
        return sdict

    keep_transforms = []
    for tdict in sdict['transforms']:
        cls = SERIALIZABLE_REGISTRY[tdict['__class_fullname__']]
        if 'transforms' in tdict:
            keep_transforms.append(remove_image_only_transforms(tdict))
        elif not issubclass(cls, ImageOnlyTransform):
            keep_transforms.append(tdict)
    sdict['transforms'] = keep_transforms

    return sdict


def elastic_distortion(images, grid_width=None, grid_height=None, magnitude=8, rs=None):
    """
    Elastic distortion operation from the Augmentor library
    Source:
    https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py
    Distorts the passed image(s) according to the parameters supplied during
    instantiation, returning the newly distorted image.
    :param images: The image(s) to be distorted.
    :type images: List containing.
    :return: List of transformed images.
    """

    if rs is None:
        rs = np.random.RandomState()
    if grid_width is None:
        grid_width = rs.randint(4,8)
    if grid_height is None:
        grid_height = rs.randint(4,8)

    extra_dim = [False] * len(images)
    redundant_dims = [False] * len(images)
    dtypes = [img.dtype for img in images]

    # Convert numpy arrays to PIL images
    for i, img in enumerate(images):
        if img.ndim == 3 and img.shape[2] > 1:
            redundant_dims[i] = True
            img = img[:, :, 0]
        elif img.ndim == 3:
            extra_dim[i] = True
        if img.dtype != np.uint8:
            max = np.max(img)
            dtype = np.uint8 if max <= 255 else np.uint16
            img = img*255 if max == 1 else img
            img = img.astype(dtype)
        mode = "L" if img.dtype == np.uint8 else "I"
        images[i] = Image.fromarray(np.squeeze(img), mode = mode)

    w, h = images[0].size

    horizontal_tiles = grid_width
    vertical_tiles = grid_height

    width_of_square = int(floor(w / float(horizontal_tiles)))
    height_of_square = int(floor(h / float(vertical_tiles)))

    width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
    height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

    dimensions = []

    for vertical_tile in range(vertical_tiles):
        for horizontal_tile in range(horizontal_tiles):
            if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                dimensions.append([horizontal_tile * width_of_square,
                                    vertical_tile * height_of_square,
                                    width_of_last_square + (horizontal_tile * width_of_square),
                                    height_of_last_square + (height_of_square * vertical_tile)])
            elif vertical_tile == (vertical_tiles - 1):
                dimensions.append([horizontal_tile * width_of_square,
                                    vertical_tile * height_of_square,
                                    width_of_square + (horizontal_tile * width_of_square),
                                    height_of_last_square + (height_of_square * vertical_tile)])
            elif horizontal_tile == (horizontal_tiles - 1):
                dimensions.append([horizontal_tile * width_of_square,
                                    vertical_tile * height_of_square,
                                    width_of_last_square + (horizontal_tile * width_of_square),
                                    height_of_square + (height_of_square * vertical_tile)])
            else:
                dimensions.append([horizontal_tile * width_of_square,
                                    vertical_tile * height_of_square,
                                    width_of_square + (horizontal_tile * width_of_square),
                                    height_of_square + (height_of_square * vertical_tile)])

    last_column = []
    for i in range(vertical_tiles):
        last_column.append((horizontal_tiles-1)+horizontal_tiles*i)

    last_row = range(horizontal_tiles * vertical_tiles - horizontal_tiles,
                     horizontal_tiles * vertical_tiles)

    polygons = []
    for x1, y1, x2, y2 in dimensions:
        polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

    polygon_indices = []
    for i in range((vertical_tiles * horizontal_tiles) - 1):
        if i not in last_row and i not in last_column:
            polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

    for a, b, c, d in polygon_indices:
        dx = rs.randint(-magnitude, magnitude)
        dy = rs.randint(-magnitude, magnitude)

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
        polygons[a] = [x1, y1,
                        x2, y2,
                        x3 + dx, y3 + dy,
                        x4, y4]

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
        polygons[b] = [x1, y1,
                        x2 + dx, y2 + dy,
                        x3, y3,
                        x4, y4]

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
        polygons[c] = [x1, y1,
                        x2, y2,
                        x3, y3,
                        x4 + dx, y4 + dy]

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
        polygons[d] = [x1 + dx, y1 + dy,
                        x2, y2,
                        x3, y3,
                        x4, y4]

    generated_mesh = []
    for i, dim in enumerate(dimensions):
        generated_mesh.append([dim, polygons[i]])

    def do_transform(image):
        return image.transform(image.size, Image.MESH, generated_mesh, resample=Image.BICUBIC)

    augmented_images = []

    for image in images:
        augmented_images.append(do_transform(image))

    for i, augmented_img in enumerate(augmented_images):
        # Convert PIL image back to numpy array
        augmented_images[i] = np.asarray(augmented_img).astype(dtypes[i])
        if extra_dim[i]:
            augmented_images[i] = np.expand_dims(augmented_images[i], axis=2)
        elif redundant_dims[i]:
            augmented_images[i] = np.repeat(augmented_images[i][:, :, np.newaxis], 3, axis=2)

    return augmented_images

def rand_square_crop(image, mask, size_options=None, rs=None):
    if rs is None:
        rs = np.random.RandomState()
    if size_options is None:
        size_options = [128, 256, 512, min(image.shape[:2])]
    crop_size = rs.choice(size_options)
    crop_start_x_max = image.shape[1] - crop_size
    crop_start_y_max = image.shape[0] - crop_size
    crop_start_x = rs.randint(0, crop_start_x_max+1)
    crop_start_y = rs.randint(0, crop_start_y_max+1)
    image = image[crop_start_y:crop_start_y+crop_size, crop_start_x:crop_start_x+crop_size]
    mask = mask[crop_start_y:crop_start_y+crop_size, crop_start_x:crop_start_x+crop_size]
    return image, mask

def remove_small_objects_in_mask(mask, min_size):
    labeled_regions = measure.label(mask[:,:,0], connectivity=1)
    labeled_regions = morphology.remove_small_objects(labeled_regions, min_size=min_size)
    mask[:,:,0][labeled_regions == 0] = 0

    return mask

def resize_image_mask(image, mask, new_shape):
    if image.shape[:2] == new_shape:
        return image, mask

    img_interpolation = cv2.INTER_LINEAR if image.shape[0] < new_shape[0] else cv2.INTER_AREA
    image = cv2.resize(image, new_shape, interpolation=img_interpolation)
    mask = cv2.resize(mask, new_shape, interpolation=cv2.INTER_NEAREST_EXACT)

    return image, mask
