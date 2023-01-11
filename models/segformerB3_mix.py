from functools import partial
import torch

from easydict import EasyDict as edict
import albumentations as A
import yaml

from isegm.model.losses import NormalizedFocalLossSigmoid, WFNL
from isegm.model.metrics import AdaptiveIoU
from isegm.data.points_sampler import MultiPointSampler
from isegm.model.is_segformer_model import SegFormerModel
from isegm.data.datasets.tubes import TubesDataset
from isegm.data.compose import ProportionalComposeDataset
from isegm.engine.focalclick_trainer import ISTrainer

MODEL_NAME = 'segformerB3_tubes'
min_object_area = 50

def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)

def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (512, 512)
    model_cfg.num_max_points = 24

    model = SegFormerModel(pipeline_version = 's2', model_version = 'b3', use_leaky_relu=True,
                           use_rgb_conv=False, use_disks=True, norm_radius=5,
                           binary_prev_mask=False, with_prev_mask=True, with_aux_output=True)

    if not cfg.distributed:
        model.to(cfg.device)
    else:
        model = model.cuda()
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.feature_extractor.load_pretrained_weights(cfg.pretrained_weights)

    return model, model_cfg

def train(model, cfg, model_cfg):
    cfg.batch_size = 28 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0

    loss_cfg.instance_refine_loss = WFNL(alpha=0.5, gamma=2, w=0.5)
    loss_cfg.instance_refine_loss_weight = 1.0

    loss_cfg.trimap_loss = torch.nn.BCEWithLogitsLoss()
    loss_cfg.trimap_loss_weight = 1.0

    min_object_area = 150

    points_sampler = MultiPointSampler(model_cfg.num_max_points, prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2,
                                       use_hierarchy=False,
                                       first_click_center=True)

    custom_augmentor = {
        'after': [
            ("elastic_distortion", []),
            ("remove_small_objects", [min_object_area])
        ]
    }

    train_augmentor = A.Compose([
        A.RandomCrop(*model_cfg.crop_size),
        A.Flip(),
        A.RandomRotate90(),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.25),
            A.RandomToneCurve(scale=0.75, p=0.75)
        ], p=0.75)
    ])

    val_augmentor = A.Compose([
        A.RandomCrop(*model_cfg.crop_size),
        A.Flip(),
        A.RandomRotate90(),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.RandomToneCurve(p=0.5)
        ], p=0.75)
    ])

    if cfg.dataset_path.endswith('.yml'):
        train_datasets = []
        val_datasets = []
        ratios = []
        with open(cfg.dataset_path, 'r') as f:
            dataset_paths_and_ratios = yaml.safe_load(f)

        sum_ratios = sum(dataset_paths_and_ratios.values())
        for dataset_path, ratio in dataset_paths_and_ratios.items():
            train_datasets.append(TubesDataset(dataset_path=dataset_path, split='train',
                                             custom_augmentor=custom_augmentor))
            val_datasets.append(TubesDataset(dataset_path=dataset_path, split='val',
                                             custom_augmentor=custom_augmentor))
            ratios.append(ratio / sum_ratios)

        trainset = ProportionalComposeDataset(
            datasets=train_datasets,
            ratios=ratios,
            augmentator=train_augmentor,
            min_object_area=min_object_area,
            points_sampler=points_sampler,
            epoch_len=2000
        )
        valset = ProportionalComposeDataset(
            datasets=val_datasets,
            ratios=ratios,
            augmentator=val_augmentor,
            min_object_area=min_object_area,
            points_sampler=points_sampler,
            epoch_len=400
        )
    else:
        trainset = TubesDataset(
            dataset_path=cfg.dataset_path,
            split='train',
            custom_augmentor=custom_augmentor,
            augmentator=train_augmentor,
            min_object_area=min_object_area,
            points_sampler=points_sampler,
            epoch_len=2000
        )
        valset = TubesDataset(
            cfg.dataset_path,
            split='val',
            custom_augmentor=custom_augmentor,
            augmentator=val_augmentor,
            min_object_area=min_object_area,
            points_sampler=points_sampler,
            epoch_len=400
        )

    optimizer_params = {
        'lr': 5e-3, 'betas': (0.9, 0.999), 'eps': 1e-8
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[190, 210], gamma=0.1)
    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=[(0, 50), (200, 5)],
                        image_dump_interval=500,
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=3)
    trainer.run(num_epochs=230)
