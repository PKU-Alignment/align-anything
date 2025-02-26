# Get path of the detic repo
import os
import sys

try:
    DETIC_REPO_PATH = next(p for p in sys.path if p.endswith("Detic") or p.endswith("Detic/"))
except StopIteration:
    try:
        DETIC_REPO_PATH = os.getenv("DETIC_REPO_PATH")
    except KeyError:
        raise ImportError("Could not find Detic repo path. Please add Detic to your PYTHONPATH.")

# Add CenterNet2 to the path
try:
    import centernet
except ImportError:
    _center_net_path = os.path.join(DETIC_REPO_PATH, "third_party/CenterNet2")
    if not os.path.exists(_center_net_path):
        raise ImportError(f"Path {_center_net_path} does not exist")

    sys.path.insert(0, _center_net_path)


import os
from typing import Optional, Sequence, List, TYPE_CHECKING

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.modeling import build_model
from torch.distributions.utils import lazy_property
from torch.nn import functional as F
from torchvision.transforms import Resize

# from data_generation.object_detection import DETIC_REPO_PATH
sys.path.insert(0, DETIC_REPO_PATH)
from detic.config import add_detic_config

if TYPE_CHECKING:
    # Assumes you've set up Detic to be in your IDE's PYTHONPATH
    from third_party.CenterNet2.centernet.config import add_centernet_config
else:
    try:
        from centernet.config import add_centernet_config
    except ImportError:
        raise ImportError("Please set up your python path to include Detic/third_party/CenterNet2")


def create_detic_cfg(
    config_file: str,
    opts: Optional[List[str]],
    confidence_threshold: float,
    pred_all_class: bool,
    device: str,
):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = (
        device
        if isinstance(device, (str, int))
        else ("cpu" if device.index is None else device.index)
    )
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"  # load later
    if not pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True

    cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = os.path.join(
        DETIC_REPO_PATH, cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH
    )

    cfg.freeze()
    return cfg


def resize_boxes(boxes, original_size, new_size, cutoff_amount=6):
    """
    Resize bounding boxes from original image size to new image size.

    Args:
    - boxes (list of lists): List of bounding boxes in the format [x1, y1, x2, y2].
    - original_size (tuple): Original image size in the format (original_height, original_width).
    - new_size (tuple): New image size in the format (new_height, new_width).

    Returns:
    - list of lists: Resized bounding boxes.
    """

    original_height, original_width = original_size
    new_height, new_width = new_size

    scale_x = new_width / original_width
    scale_y = new_height / original_height

    resized_boxes = []

    for box in boxes:
        x1, y1, x2, y2 = box
        resized_x1 = int(x1 * scale_x) - cutoff_amount  # woof
        resized_y1 = int(y1 * scale_y)
        resized_x2 = int(x2 * scale_x) - cutoff_amount
        resized_y2 = int(y2 * scale_y)
        resized_boxes.append([resized_x1, resized_y1, resized_x2, resized_y2])

    return resized_boxes


class DeticPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a batch of input images.

    Note:
    1. Always assume you are given a batch of RGB images as input of shape B x H x W x 3.
    2. Will apply resizing defined by min_size_test/max_size_test
    """

    def __init__(
        self,
        vocabulary: Sequence[str] = ("apple", "potato"),
        prompt: str = "a ",
        config_file: str = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        model_weights_file: str = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
        min_size_test: Optional[int] = None,
        max_size_test: Optional[int] = None,
        confidence_threshold=0.3,
        pred_all_class=False,
        device="cpu",
    ):
        if not os.path.exists(config_file):
            config_file = os.path.join(DETIC_REPO_PATH, "configs", config_file)
            assert os.path.exists(config_file)

        if not os.path.exists(model_weights_file):
            model_weights_file = os.path.join(DETIC_REPO_PATH, "models", model_weights_file)
            assert os.path.exists(model_weights_file)

        opts = [
            "MODEL.WEIGHTS",
            model_weights_file,
        ]

        if min_size_test is not None:
            opts.extend(["INPUT.MIN_SIZE_TEST", min_size_test])

        if max_size_test is not None:
            opts.extend(["INPUT.MAX_SIZE_TEST", max_size_test])

        cfg = create_detic_cfg(
            config_file=config_file,
            opts=opts,
            confidence_threshold=confidence_threshold,
            pred_all_class=pred_all_class,
            device=device,
        )

        self.cfg = cfg.clone()  # cfg can be modified by model
        self.prompt = prompt

        self.model = build_model(self.cfg)
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.model.eval()

        self._vocabulary: Optional[str] = None
        self.vocabulary = vocabulary

        assert cfg.INPUT.FORMAT == "RGB"

    def to(self, device: torch.device):
        self.model.to(device)
        self.text_encoder.to(device)
        return self

    @property
    def vocabulary(self) -> Sequence[str]:
        return self._vocabulary

    @lazy_property
    def text_encoder(self):
        from detic.modeling.text.text_encoder import build_text_encoder

        text_encoder = build_text_encoder(pretrain=True)
        text_encoder.eval()
        text_encoder.to(self.model.device)
        return text_encoder

    def get_clip_embeddings(self, vocabulary, prompt="a "):
        texts = [prompt + x for x in vocabulary]
        with torch.no_grad():
            return self.text_encoder(texts).detach().permute(1, 0).contiguous()

    @vocabulary.setter
    def vocabulary(self, vocabulary: Sequence[str]):
        if self._vocabulary is not None and list(self._vocabulary) == list(vocabulary):
            return
        self._vocabulary = vocabulary

        num_classes = len(self._vocabulary)

        self.model.roi_heads.num_classes = num_classes

        zs_weight = self.get_clip_embeddings(self._vocabulary, prompt=self.prompt)
        zs_weight = torch.cat(
            [zs_weight, zs_weight.new_zeros((zs_weight.shape[0], 1), device=self.model.device)],
            dim=1,
        )  # D x (C + 1)

        if self.model.roi_heads.box_predictor[0].cls_score.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, dim=0)

        for k in range(len(self.model.roi_heads.box_predictor)):
            del self.model.roi_heads.box_predictor[k].cls_score.zs_weight
            self.model.roi_heads.box_predictor[k].cls_score.zs_weight = zs_weight

    def resize_images(self, images: torch.Tensor):
        """
        Resize images to the target size.
        """
        b, c, h, w = images.shape
        new_h, new_w = ResizeShortestEdge.get_output_shape(
            oldh=h,
            oldw=w,
            short_edge_length=self.cfg.INPUT.MIN_SIZE_TEST,
            max_size=self.cfg.INPUT.MAX_SIZE_TEST,
        )

        return Resize((new_h, new_w), antialias=True)(images)

    def __call__(self, images: torch.Tensor):
        """
        Args:
            original_image (np.ndarray): an image of shape (B x C x H x W) (in RGB order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            nbatch, _, height, width = images.shape
            images = self.resize_images(images)
            images = images.float()

            inputs = []
            for i in range(nbatch):
                inputs.append({"image": images[i], "height": height, "width": width})

            predictions = self.model(inputs)
            return predictions
