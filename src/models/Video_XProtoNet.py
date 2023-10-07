import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.receptive_field import compute_proto_layer_rf_info_v2
from src.models.ProtoPNet import PPNet, base_architecture_to_features


class Video_XProtoNet(PPNet):
    def __init__(
        self, cnn_backbone, img_size, prototype_shape, proto_layer_rf_info, num_classes, init_weights=True, **kwargs
    ):
        # super(Video_XProtoNet, self).__init__(**kwargs)
        super(PPNet, self).__init__()  # not calling init of PPNet and directly going to its parent!

        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.prototype_class_identity = self.get_prototype_class_identity()
        self.proto_layer_rf_info = proto_layer_rf_info

        # CNN Backbone module
        self.cnn_backbone = cnn_backbone
        cnn_backbone_out_channels = self.get_cnn_backbone_out_channels(self.cnn_backbone)

        # feature extractor module
        self.add_on_layers = nn.Sequential(
            nn.Conv3d(
                in_channels=cnn_backbone_out_channels,
                out_channels=self.prototype_shape[1],
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=self.prototype_shape[1],
                out_channels=self.prototype_shape[1],
                kernel_size=1,
            ),
        )

        # Occurrence map module
        self.occurrence_module = nn.Sequential(
            nn.Conv3d(
                in_channels=cnn_backbone_out_channels,
                out_channels=self.prototype_shape[1],
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=self.prototype_shape[1],
                out_channels=self.prototype_shape[1] // 2,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=self.prototype_shape[1] // 2,
                out_channels=self.prototype_shape[0],
                kernel_size=1,
                bias=False,
            ),
            # nn.Conv3d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[0], kernel_size=1, bias=False),
        )

        self.om_softmax = nn.Softmax(dim=-1)
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

        # Learnable prototypes
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)

        # To be used for pruning
        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)  # do not use bias

        if init_weights:
            self._initialize_weights(self.add_on_layers)
            self._initialize_weights(self.occurrence_module)
            self.set_last_layer_incorrect_connection(incorrect_strength=0)

    def forward(self, x):
        # Feature Extractor Layer
        x = self.cnn_backbone(x)  # shape (N, 512 or 256, T, H, W)
        feature_map = self.add_on_layers(x).unsqueeze(1)  # shape (N, 1, D, T, H, W)
        occurrence_map = self.get_occurence_map_absolute_val(x)  # shape (N, P, 1, T, H, W)
        features_extracted = (occurrence_map * feature_map).sum(dim=3).sum(dim=3).sum(dim=3)  # shape (N, P, D)

        # Prototype Layer
        similarity = self.cosine_similarity(
            features_extracted, self.prototype_vectors.squeeze().unsqueeze(0)
        )  # shape (N, P)
        similarity = (similarity + 1) / 2.0  # normalizing to [0,1] for positive reasoning

        # classification layer
        logits = self.last_layer(similarity)

        return logits, similarity, occurrence_map

    def compute_occurence_map(self, x):
        # Feature Extractor Layer
        x = self.cnn_backbone(x)
        occurrence_map = self.get_occurence_map_absolute_val(x)  # shape (N, P, 1, T, H, W)
        return occurrence_map

    def get_occurence_map_absolute_val(self, x):
        occurrence_map = self.occurrence_module(x)  # shape (N, P, T, H, W)
        occurrence_map = torch.abs(occurrence_map).unsqueeze(2)  # shape (N, P, 1, T, H, W)
        return occurrence_map

    def push_forward(self, x):
        """
        this method is needed for the pushing operation
        """
        # Feature Extractor Layer
        x = self.cnn_backbone(x)  # shape (N, 512 or 256, T, H, W)
        feature_map = self.add_on_layers(x).unsqueeze(1)  # shape (N, 1, D, T, H, W)
        occurrence_map = self.get_occurence_map_absolute_val(x)  # shape (N, P, 1, T, H, W)
        features_extracted = (occurrence_map * feature_map).sum(dim=3).sum(dim=3).sum(dim=3)  # shape (N, P, D)

        # Prototype Layer
        similarity = self.cosine_similarity(
            features_extracted, self.prototype_vectors.squeeze().unsqueeze(0)
        )  # shape (N, P)
        similarity = (similarity + 1) / 2.0  # normalizing to [0,1] for positive reasoning

        # classification layer
        logits = self.last_layer(similarity)

        return features_extracted, 1 - similarity, occurrence_map, logits

    def __repr__(self):
        # XProtoNet(self, backbone, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            "PPNet(\n"
            "\tcnn_backbone: {},\n"
            "\timg_size: {},\n"
            "\tprototype_shape: {},\n"
            "\tproto_layer_rf_info: {},\n"
            "\tnum_classes: {},\n"
            ")"
        )

        return rep.format(
            self.cnn_backbone,
            self.img_size,
            self.prototype_shape,
            self.proto_layer_rf_info,
            self.num_classes,
        )


def construct_Video_XProtoNet(
    base_architecture,
    pretrained=True,
    img_size=224,
    prototype_shape=(40, 256, 1, 1, 1),
    num_classes=4,
    backbone_last_layer_num=-3,
):
    cnn_backbone = base_architecture_to_features[base_architecture](
        pretrained=pretrained, last_layer_num=backbone_last_layer_num
    )
    # layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    # proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
    #                                                      layer_filter_sizes=layer_filter_sizes,
    #                                                      layer_strides=layer_strides,
    #                                                      layer_paddings=layer_paddings,
    #                                                      prototype_kernel_size=prototype_shape[2])
    return Video_XProtoNet(
        cnn_backbone=cnn_backbone,
        img_size=img_size,
        prototype_shape=prototype_shape,
        proto_layer_rf_info=None,
        num_classes=num_classes,
        init_weights=True,
    )
