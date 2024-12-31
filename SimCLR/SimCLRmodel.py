import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features
        
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        if model_name in self.resnet_dict:
            model = self.resnet_dict[model_name]
            print(f"Feature extractor: {model_name}")
            return model
        else:
            raise ValueError("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()  # 배치 차원 제외

        # projection MLP
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x
