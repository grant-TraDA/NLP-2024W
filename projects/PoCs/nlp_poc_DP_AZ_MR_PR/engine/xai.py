from typing import Callable

import torch
from captum.attr import FeatureAblation
from torch import Tensor


class FeatureAblationText:

    def __init__(
        self,
        model: Callable[[Tensor], Tensor],
        preprocessing_fn: Callable[[any], Tensor] | None = None,
    ) -> None:

        self.model = model

        if preprocessing_fn:
            self.preprocessing_fn = preprocessing_fn
        else:
            self.preprocessing_fn = lambda x: x

        self.attr = FeatureAblation(model)

    def get_attributions(self, obs: list[any]) -> Tensor:
        obs = self.preprocessing_fn(obs)
        exps = [None for _ in range(len(obs))]
        for idx, ob in enumerate(obs):
            exp = self.attr.attribute(ob)
            exps[idx] = exp

        return torch.stack(exps)

    def get_grouped_attribution(
        self, obs: list[any], masks: list[Tensor]
    ) -> Tensor:
        obs = self.preprocessing_fn(obs)
        exps = []
        for ob, mask in zip(obs, masks):
            exp = self.attr.attribute(ob, feature_mask=mask)
            exps.append(exp)
        return torch.stack(exps)
