from contextlib import contextmanager
from typing import Dict, List
import torch.nn as nn
import torch.nn.functional as F
import torch


class ComponentModeller(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.BatchNorm1d(in_channels // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_channels // 2, in_channels // 2),
            nn.BatchNorm1d(in_channels // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_channels // 2, in_channels // 2),
            nn.BatchNorm1d(in_channels // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_channels // 2, in_channels),
        )
        self.mix_factor = nn.Linear(in_channels, in_channels)

        self.discriminator = nn.Sequential(
            nn.Linear(in_channels, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, feats: torch.Tensor):
        feats = feats.detach()
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).squeeze(3).squeeze(2)

        embeddings = self.encoder(feats)

        mix_factor = self.mix_factor(embeddings).sigmoid()

        set_info = feats * mix_factor
        set_preds = self.discriminator(set_info)

        class_info = feats - set_info
        class_preds = self.classifier(class_info)

        return set_preds, class_preds, mix_factor


class DACRegularizer(nn.Module):
    def __init__(self, num_classes: int, layer_configs: List[Dict], alpha: float = 3, beta: float = 0.1):
        super().__init__()

        self.alpha = alpha  # useless
        self.beta = beta

        self.component_modellers = nn.ModuleDict()

        self.layer_names = []
        for layer_config in layer_configs:
            layer_name = layer_config["name"].replace(".", "_")
            layer_channels = layer_config["channels"]

            self.component_modellers[layer_name] = ComponentModeller(layer_channels, num_classes)

            self.layer_names.append(layer_name)

        self.set_preds = {}
        self.class_preds = {}

        self.activate_drop = False
        self.activate_record = False
        self.record_key = None

    def hook(self, model: nn.Module):
        for layer_name, layer in model.named_modules():
            if layer_name.replace(".", "_") in self.layer_names:
                eval(
                    f"layer.register_forward_pre_hook(lambda layer, input: self._train_hook('{layer_name.replace('.', '_')}', layer, input))",
                    {
                        "self": self,
                        "layer": layer,
                    },
                )

    def _train_hook(self, layer_name: str, layer: nn.Module, input: torch.Tensor):
        if self.activate_drop:
            self.component_modellers[layer_name].eval()
            with torch.no_grad():
                set_pred, class_pred, set_mix_factor = self.component_modellers[layer_name](input[0])

                set_mix_factor = (set_mix_factor - 0.5).abs() * 2

                sorted_set_mix_factor, sorted_set_mix_factor_idx = set_mix_factor.sort(dim=1, descending=False)
                threshold = sorted_set_mix_factor[
                    :,
                    min(int(sorted_set_mix_factor.shape[1] * self.beta), sorted_set_mix_factor.shape[1] - 1),
                ]

                ign_mask = ((set_mix_factor < self.alpha) & (set_mix_factor < threshold.unsqueeze(1))).float()

                drop_mask = ign_mask.logical_not().float()

            droped_feats = input[0] * drop_mask.view(
                [*drop_mask.shape, *[1 for _ in range(input[0].dim() - drop_mask.dim())]]
            )

            droped_feats = F.dropout2d(input[0], abs(self.beta), training=True)
            self.component_modellers[layer_name].train()
            return droped_feats
        elif self.activate_record:
            if self.record_key not in self.set_preds:
                self.set_preds[self.record_key] = {}
            if self.record_key not in self.class_preds:
                self.class_preds[self.record_key] = {}

            set_pred, class_pred, set_mix_factor = self.component_modellers[layer_name](input[0])

            self.set_preds[self.record_key][layer_name] = set_pred
            self.class_preds[self.record_key][layer_name] = class_pred

    def forward(self, labels: Dict[str, List[torch.Tensor]]):
        loss_cls = 0
        loss_set = 0

        for key, labels in labels.items():
            class_labels, set_labels = labels

            for (layer_name, set_pred), (_, class_pred) in zip(
                self.set_preds[key].items(), self.class_preds[key].items()
            ):
                set_preds = set_pred
                class_preds = class_pred

                loss_cls += F.cross_entropy(class_preds, class_labels)
                loss_set += F.binary_cross_entropy_with_logits(set_preds, set_labels.view_as(set_preds))

        return loss_cls, loss_set

    @contextmanager
    def drop(self):
        self.activate_drop = True
        yield
        self.activate_drop = False

    @contextmanager
    def record(self, recrd_key: str):
        self.activate_record = True
        self.record_key = recrd_key
        yield
        self.activate_record = False
        self.record_key = None


# =================================================================================================
# Usage:
# =================================================================================================
# if __name__ == "__main__":
#     regularizer = DACRegularizer(10, [{"name": "conv1", "channels": 3}])

#     with regularizer.record("l"):
#         model(x_l)
#     with regularizer.record("u"):
#         model(x_u)

#     with regularizer.drop():
#         y_l = model(x_l)
#         y_u = model(x_u)

#     l_cm_cls, l_cm_set = regularizer(
#         {
#             "l": [
#                 y_l_gt
#                 torch.ones_like(y_l_gt).float(),
#             ],
#             "u": [
#                 y_u_pseudo,
#                 torch.zeros_like(y_u_pseudo).float(),
#             ],
#         }
#     )

#     loss = F.cross_entropy(y_l, y_l_gt) + F.cross_entropy(y_u, y_u_pseudo) + l_cm_cls + l_cm_set
