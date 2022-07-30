from torch import dropout
from torchvision.models import swin_s, Swin_S_Weights, SwinTransformer, Swin_B_Weights
from typing import List, Optional, Union, Any, Dict
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import V


class IntSwinS(SwinTransformer):
    def forward(self, x):
        # self.x_int = [x]
        self.x_int = []
        for i in range(len(self.features)):
            x = self.features[i](x)
            if i % 2 == 1 and i < len(self.features) - 2:
                # We don't want to the last one as it is the output
                self.x_int.append(x.permute(0, 3, 1, 2))

        return x.permute(0, 3, 1, 2)


def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(
                f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value


def _swin_transformer(
    patch_size: List[int],
    embed_dim: int,
    depths: List[int],
    num_heads: List[int],
    window_size: List[int],
    stochastic_depth_prob: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> SwinTransformer:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes",
                              len(weights.meta["categories"]))

    model = IntSwinS(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def swin_s(*, weights: Optional[Swin_S_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_small architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/pdf/2103.14030>`_.
    Args:
        weights (:class:`~torchvision.models.Swin_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.Swin_S_Weights
        :members:
    """
    weights = Swin_S_Weights.verify(weights)

    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.3,
        weights=weights,
        progress=progress,
        **kwargs,
    )


def swin_b(*, weights: Optional[Swin_B_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_base architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/pdf/2103.14030>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_B_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_B_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_B_Weights
        :members:
    """
    weights = Swin_B_Weights.verify(weights)

    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[7, 7],
        stochastic_depth_prob=0.5,
        weights=weights,
        progress=progress,
        **kwargs,
    )


def swin_pretrained_s():
    weights = Swin_S_Weights.IMAGENET1K_V1
    return swin_s(weights=weights)


def swin_pretrained_b():
    weights = Swin_B_Weights.IMAGENET1K_V1
    return swin_b(weights=weights)
