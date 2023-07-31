import argparse, os
from typing import Any, Dict
import torch
import piq
from torchvision import transforms
from torch.utils.data import ConcatDataset
import tqdm
from ssl_robustness.evaluation.imagenet_c_dataloader import ImagenetCDataloader
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import pandas as pd

@torch.no_grad()
def main(args):
    device = torch.device(f"cuda{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    print("GPU being used: ", device)

    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    data = ImagenetCDataloader(args.test_dataset, transform, batch_size=100, num_workers=4, shuffle=False).datasets
    snow = [(d, k.split('_')[-1]) for k,d in data.items() if k.startswith('weather_snow')]
    elastic_transform = [(d, k.split('_')[-1]) for k,d in data.items() if k.startswith('digital_elastic_transform')]
    saturate = [(d, k.split('_')[-1]) for k,d in data.items() if k.startswith('extra_saturate')]
    
    imagenet = DataLoader(ImageFolder(args.imagenet_path, transform=transform),batch_size=100, num_workers=4, shuffle=False)
    
    def calculate_psnr_metric(targets, base, key: str, res: dict = {}) -> Dict[str, Any]:

        if key not in res:
            res[key] = {}
        for data, diff in tqdm.tqdm(targets, key):
            for a, b in zip(data, base):
                targets, _ = b
                inputs, _ = a
                
                targets = targets.to(device)
                inputs = inputs.to(device)
                
                val = piq.ssim(inputs,targets)
                val = val.mean().item()
                if diff in res[key]:
                    res[key][diff] +=  val
                    res[key][diff] /= 2
                else:
                    res[key][diff] =  val
   
                # break
        return res
    
    psnr_data = calculate_psnr_metric(snow, imagenet, "snow")
    psnr_data = calculate_psnr_metric(elastic_transform, imagenet, "elastic transform", psnr_data)
    psnr_data = calculate_psnr_metric(saturate, imagenet, "saturate", psnr_data)
    
    df = pd.DataFrame(psnr_data)
    
    if os.path.exists(f"{args.output}/statistical_trends") is not True:
        os.system("mkdir -p {}".format(f"{args.output}/statistical_trends"))
    df.to_csv(f"{args.output}/statistical_trends/ssim.csv")
    

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Compute Structural_Similarity')
    parser.add_argument('--test_dataset', type=str, default='ssl_robustness/data/imagenet_c', help='location of the ImageNet-C dataset in this version')
    parser.add_argument('--imagenet_path', type=str, default='ssl_robustness/data/imagenet', help='location of the ImageNet dataset')
    parser.add_argument('--output', type=str, default='ssl_robustness/output/', help='location of output directory for saving results')
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    
    args = parser.parse_args()
    
    main(args)