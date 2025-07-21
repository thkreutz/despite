import torch
import torch.nn as nn

class ClassifierWrapperH(nn.Module):
    def __init__(self, backbone, num_classes=10, freeze_backbone=True, probing="linear"):
        """
        Args:
            backbone: The pre-trained backbone model (with or without a projection head).
            num_classes: Number of classes for classification.
            freeze_backbone: If True, the backbone's parameters will be frozen (for linear probing).
            non_linear: If True, uses an MLP classifier instead of a single linear layer.
        """
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False  # Freeze all backbone layers

        # Define classifier
        feature_dim = 80 #backbone.output_dim  # Ensure backbone has this attribute

        if probing == "non-linear":
            self.classifier = nn.Sequential(nn.LayerNorm(feature_dim,),
                                            nn.Linear(feature_dim, 256),
                                            nn.GELU(),
                                            nn.Dropout(0.1),
                                            nn.Linear(256, num_classes))
        else:
            self.classifier = nn.Linear(feature_dim, num_classes)

        # Replace model head with our classifier
        self.backbone.head = self.classifier

    def forward(self, x):
        return self.backbone(x)  # Model already has classifier as `.head`


class ClassifierWrapperN(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes=10, freeze_backbone=True, probing="linear"):
        """
        Args:
            backbone: The pre-trained backbone model (with or without a projection head).
            num_classes: Number of classes for classification.
            freeze_backbone: If True, the backbone's parameters will be frozen (for linear probing).
            non_linear: If True, uses an MLP classifier instead of a single linear layer.
        """
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False  # Freeze all backbone layers

        # Define classifier
        #feature_dim = #backbone.output_dim  # Ensure backbone has this attribute

        if probing == "non-linear":
            self.classifier = nn.Sequential(nn.LayerNorm(feature_dim,),
                                            nn.Linear(feature_dim, 256),
                                            nn.GELU(),
                                            nn.Dropout(0.1),
                                            nn.Linear(256, num_classes))
        else:
            self.classifier = nn.Linear(feature_dim, num_classes)


    def forward(self, x):
        z = self.backbone(x)
        if type(z) == dict:
            z = z["mu"]
        return self.classifier(z)  # Model already has classifier as `.head`