import argparse
import json
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from metric import MetricPage, RobustnessErrorRate
from ssl_robustness.evaluation.imagenet_c_dataloader import ImagenetCDataloader, curruptions_subtypes
import mmengine
import tqdm

def main(args):
    device = torch.device(
        f"cuda{args.device}") if torch.cuda.is_available() else torch.device("cpu")

    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize(224),
        # transforms.CenterCrop(224),
    ])


    data = ImagenetCDataloader(
        args.test_dataset, transform,
        shuffle=True,
        num_workers=4,
        batch_size=128
    )


    def load_model(path, replace_key=''):
        state_dict = torch.load(path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            for k in list(state_dict.keys()):
                # remove prefix
                if k.startswith(replace_key):
                    state_dict[k[len(replace_key + "."):]] = state_dict[k]
                elif k.startswith("head"):
                    state_dict[k[len("head."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
        encoder = models.resnet50()
        msg = encoder.load_state_dict(state_dict, strict=False)

        assert (len(msg.missing_keys) == 0)

        for name, param in encoder.named_parameters():
            param.requires_grad = False

        encoder.eval()
        return encoder


    barlow = nn.DataParallel(load_model(
        "ssl_robustness/models/models/barlow/resnet50_linear-8xb32-coslr-100e_in1k_20220825-52fde35f.pth", "backbone"))
    barlow.to(device)
    metric_barlow = RobustnessErrorRate(100-71.8, collect_device=device)

    byol = nn.DataParallel(load_model(
        "ssl_robustness/models/models/byol/resnet50_linear-8xb512-coslr-90e_in1k_20220825-7596c6f5.pth", "backbone"))
    byol.to(device)
    metric_byol = RobustnessErrorRate(100-71.8, collect_device=device)

    simclr = nn.DataParallel(load_model(
        "ssl_robustness/models/simclr/resnet50_linear-8xb512-coslr-90e_in1k_20220825-f12c0457.pth", "backbone"))
    simclr.to(device)
    metric_simclr = RobustnessErrorRate(100-66.9, collect_device=device)

    simsam = nn.DataParallel(load_model(
        "ssl_robustness/models/simsam/resnet50_linear-8xb512-coslr-90e_in1k_20220825-f53ba400.pth", "backbone"))
    simsam.to(device)
    metric_simsam = RobustnessErrorRate(100-68.3, collect_device=device)

    swav = nn.DataParallel(load_model(
        "ssl_robustness/models/swav/resnet50_linear-8xb32-coslr-100e_in1k_20220825-80341e08.pth", "backbone"))
    swav.to(device)
    metric_swav = RobustnessErrorRate(100-70.5, collect_device=device)

    dino = nn.DataParallel(load_model("ssl_robustness/models/dino/dino.pth" ))
    dino.to(device)
    metric_dino = RobustnessErrorRate(100-75.3, collect_device=device)

    
    result = {}

    for key, dataset in tqdm.tqdm(data.datasets.items()):
        for inputs, labels in dataset:
            inputs = inputs.to(device)
            labels = labels.to(device)
            cs = key.split('_')
            s = cs[-1]
            c = "_".join(cs[1:-1])

            metric_barlow.process({'input': inputs}, MetricPage(
                barlow(inputs), labels, int(s), c).list())

            metric_byol.process({'input': inputs}, MetricPage(
                byol(inputs), labels, int(s), c).list())

            metric_simclr.process({'input': inputs}, MetricPage(
                simclr(inputs), labels, int(s), c).list())

            metric_simsam.process({'input': inputs}, MetricPage(
                simsam(inputs), labels, int(s), c).list())

            metric_swav.process({'input': inputs}, MetricPage(
                swav(inputs), labels, int(s), c).list())
            
            metric_dino.process({'input': inputs}, MetricPage(
                dino(inputs), labels, int(s), c).list())
            
            

    result['barlow'] = metric_barlow.compute_metrics(metric_barlow.results)
    result['byol'] = metric_byol.compute_metrics(metric_byol.results)
    result['simclr'] = metric_simclr.compute_metrics(metric_simclr.results)
    result['simsam'] = metric_simsam.compute_metrics(metric_simsam.results)
    result['swav'] = metric_swav.compute_metrics(metric_swav.results)
    result['dino'] = metric_dino.compute_metrics(metric_dino.results)
    
    if os.path.exists(f'{args.output}/qunatitative_analysis') is not True:
        os.system("mkdir -p {}".format(f'{args.output}/qunatitative_analysis'))
    with open(f'{args.output}/qunatitative_analysis/out_ssl_methods.json', 'w') as fp:
        json.dump(result, fp)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Evaluate Mean Currption Error for all SSL methods')
    parser.add_argument('--test_dataset', type=str, default='ssl_robustness/data/imagenet_c', help='location of the ImageNet-C dataset in this version')
    parser.add_argument('--output', type=str, default='ssl_robustness/output/', help='location of output directory for saving results')
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    
    args = parser.parse_args()
    main(args)