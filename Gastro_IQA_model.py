import torch
from torchvision.models import resnet50
from torchsummary import summary


class GastroIQA(torch.nn.Module):
    def __init__(self, GastroWeights):
        super(GastroIQA, self).__init__()
        self.weights = torch.load(GastroWeights, weights_only=True)
        self.resnet = torch.nn.Sequential()
        [self.resnet.add_module(name, child) for name, child in resnet50(weights=None).named_children() if
         name != 'fc']
        self.resnet.load_state_dict(self.weights, strict=True)
        #create one head
        self.avg_pooling = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten())
        self.linear_layers = torch.nn.Linear(2048, 1)

    def forward(self, x):
        y = self.resnet(x)
        y_embeddings = self.avg_pooling(y)
        #one head
        y = self.linear_layers(y_embeddings)
        return y, y_embeddings


class GastroResnet(torch.nn.Module):
    def __init__(self, GastroWeights):
        super(GastroResnet, self).__init__()
        self.weights = torch.load(GastroWeights, weights_only=True)
        self.resnet = torch.nn.Sequential()
        [self.resnet.add_module(name, child) for name, child in resnet50(weights=None).named_children() if
         name != 'fc']
        self.resnet.load_state_dict(self.weights, strict=True)

        self.avg_pooling = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten())


    def forward(self, x):
        y = self.resnet(x)
        y_embeddings = self.avg_pooling(y)
        return y_embeddings


class GastroIQA_multihead(torch.nn.Module):
    def __init__(self, GastroWeights):
        super(GastroIQA_multihead, self).__init__()
        self.weights = torch.load(GastroWeights, weights_only=True)
        self.resnet = torch.nn.Sequential()
        [self.resnet.add_module(name, child) for name, child in resnet50(weights=None).named_children() if
         name != 'fc']
        self.resnet.load_state_dict(self.weights, strict=True)
        # create one head
        self.avg_pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.iqa_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 1)
        )
        self.esophagus_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 1)
        )

        self.cleaning_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 1)
        )

    def forward(self, x):
        y = self.resnet(x)

        #one head
        y = self.avg_pooling(y)

        #iqa
        y_iqa = self.iqa_layers(y)

        #esophagus
        y_eso = self.esophagus_layers(y)

        #cleaning
        y_cleaning = self.cleaning_layers(y)


        return y_iqa, y_eso, y_cleaning, y

class GastroIQA_multihead_2(torch.nn.Module):
    def __init__(self, GastroWeights):
        super(GastroIQA_multihead_2, self).__init__()
        self.weights = torch.load(GastroWeights, weights_only=True)
        self.resnet = torch.nn.Sequential()
        [self.resnet.add_module(name, child) for name, child in resnet50(weights=None).named_children() if
         name != 'fc']
        self.resnet.load_state_dict(self.weights, strict=True)
        # create one head
        self.avg_pooling = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 128))
        self.iqa_layers = torch.nn.Linear(128, 1)
        self.esophagus_layers = torch.nn.Linear(128, 1)
        self.cleaning_layers = torch.nn.Linear(128, 1)


    def forward(self, x):
        y = self.resnet(x)

        #one head
        y = self.avg_pooling(y)

        #iqa
        y_iqa = self.iqa_layers(y)

        #esophagus
        y_eso = self.esophagus_layers(y)

        #cleaning
        y_cleaning = self.cleaning_layers(y)


        return y_iqa, y_eso, y_cleaning