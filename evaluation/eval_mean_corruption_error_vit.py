import argparse, os
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import tqdm
from ssl_robustness.evaluation.imagenet_c_dataloader import ImagenetCDataloader

from metric import MetricPage, RobustnessErrorRate
import pandas as pd


class LinearClassifier(nn.Module):
    
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)
    
    

def main(args):
    device = torch.device( f"cuda{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    
    print("GPU being used: ", device)
    
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ])

    data = ImagenetCDataloader(
        args.imagenet_c_path, transform,
        shuffle=True,
        num_workers=4,
        batch_size=20
    )

    
    vits8: nn.Module = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    backbone = torch.load("ssl_robustness/models/dino/dino_deitsmall8_pretrain.pth", map_location="cpu")
    fc = torch.load("ssl_robustness/models/dino/dino_deitsmall8_linearweights.pth", map_location="cpu")['state_dict']
    for k in list(fc.keys()):
            # remove prefix
            if k.startswith("module"):
                fc[k[len("module" + "."):]] = fc[k]
                
            del fc[k]
    msg = vits8.load_state_dict(backbone, strict=False)
    print(msg.missing_keys)
    linear = LinearClassifier(384*4)
    linear.load_state_dict(fc, strict=False)
    msg = linear.load_state_dict(fc, strict=False)
    print(msg.missing_keys)
    

    # vits8 = nn.DataParallel(vits8)
    vits8 = vits8.to(device)
    linear = nn.DataParallel(linear)
    linear = linear.to(device)
    metric_dino = RobustnessErrorRate(100-79.7, collect_device=device)
    
    for key, dataset in tqdm.tqdm(data.datasets.items()):
        for inputs, labels in dataset:
            inputs = inputs.to(device)
            labels = labels.to(device)
            cs = key.split('_')
            s = cs[-1]
            c = "_".join(cs[1:-1])

            intermediate_output = vits8.get_intermediate_layers(inputs, 4)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            output = linear(output)
            
            metric_dino.process({'input': inputs}, MetricPage(
            output, labels, int(s), c).list())
            
        
    results = pd.DataFrame(metric_dino.compute_metrics(metric_dino.results))
    if os.path.exists(f'{args.output}/qunatitative_analysis') is not True:
        os.system("mkdir -p {}".format(f'{args.output}/qunatitative_analysis'))
    results.to_json(f"{args.output}/qunatitative_analysis/vit_dino.json")
            
            
            
                
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Evaluate Mean Currption Error for ViT (Transformer backbone)')
    parser.add_argument('--test_dataset', type=str, default='ssl_robustness/data/imagenet_c', help='location of the ImageNet-C dataset in this version')
    parser.add_argument('--output', type=str, default='ssl_robustness/output/', help='location of output directory for saving results')
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    
    args = parser.parse_args()
    main(args)
