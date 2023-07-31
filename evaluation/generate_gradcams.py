
import argparse, os
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.utils.data as utils_data
import torch.nn as nn
import tqdm
from gradcam_util import GradcamModule
from ssl_robustness.evaluation.imagenet_c_dataloader import ImagenetCDataloader
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import torch
import numpy as np

def load_model(path, device, replace_key=''):
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

    # for _, param in encoder.named_parameters():
    #     param.requires_grad = False

    encoder.eval()
    encoder.to(device)
    return encoder


def GenerateImages(path: str, gradcam):
    def decorator(func):
        def wrapper(*args, **kwargs):
            buffs = func(*args, **kwargs)
            gradcam.save_report()
            if os.path.exists("ssl_robustness/output/qualitative_analysis") is not True:
                os.system("mkdir -p {}".format("ssl_robustness/output/qualitative_analysis"))
            for name, dname, buff in buffs:
                with open(f"ssl_robustness/output/qualitative_analysis/{path}/{name}/{dname}.png", "wb+") as fp:
                    fp.write(buff.getbuffer())
            return buffs
        return wrapper
    return decorator


@GenerateImages("full_samples")
def generate_full_samples(args, data: Dict[str, DataLoader], models_list: List[Tuple[str, Any]], gradcam, transform) -> List[Tuple[str, str, BytesIO]]:

    buffs: List[Tuple[str, str, BytesIO]] = []

    for dname, dataset in tqdm.tqdm(data.items()):
        for inputs in dataset:
            for name, model in models_list:
                buff = gradcam.gradcam_on_batch_with_target(
                    model, inputs)
                buffs.append((name, dname, buff))
            break
    return buffs


def subsample_classes(classes: List[int], size: int, data: DataLoader, indexes=[]):
    if len(indexes) == 0:
        perms = torch.randint(0, len(classes), size=(size,))
        choises = torch.tensor([classes[int(i.item())] for i in perms])
        indexes = choises * 50 + torch.randint(0, 50, size=(size,))
        indexes = indexes.numpy().astype(np.int)
    vals = []
    labs = []
    for index in indexes:
        val, lab = data.dataset[index]
        vals.append(val.unsqueeze(0))
        labs.append(lab)

    vals = torch.concat(vals)
    labs = torch.tensor(labs)
    res = (vals, labs)
    return res, indexes


@GenerateImages("randomly_smapled_set")
def generate_randomly_smapled_set(args, data: Dict[str, DataLoader], models_list, gradcam, transform):
    
    imagenet = DataLoader(torchvision.datasets.ImageFolder(
        f'{args.imagenet_path}/val', transform), shuffle=True, batch_size=10)
    dog_range = list(range(151, 275))
    rep_range = list(range(29, 68))

    dogs_img, dogs_i = subsample_classes(dog_range, 5, imagenet)
    reptiles_img, reps_i = subsample_classes(rep_range, 5, imagenet)

    # == Datasets ==
    # -- saturate
    sat = [(k, DataLoader(d.dataset, shuffle=True, batch_size=10)) for k, d in data.items(
    ) if k.startswith('extra_saturate')]

    # -- elastic transform
    elst = [(k, DataLoader(d.dataset, shuffle=True, batch_size=10)) for k, d in data.items(
    ) if k.startswith('digital_elastic_transform')]

    # -- snow
    snow = [(k, DataLoader(d.dataset, shuffle=True, batch_size=10)) for k, d in data.items(
    ) if k.startswith('weather_snow')]

    # -- glass blur
    glsblr = [(k, DataLoader(d.dataset, shuffle=True, batch_size=10)) for k, d in data.items(
    ) if k.startswith('blur_glass_blur')]

    # == dogs ==
    dogs_sat = [(k, subsample_classes(dog_range, 5, d, indexes=dogs_i)[0])
                for k, d in sat]
    dogs_elst = [(k, subsample_classes(dog_range, 5, d, indexes=dogs_i)[0])
                 for k, d in elst]
    dogs_snow = [(k, subsample_classes(dog_range, 5, d, indexes=dogs_i)[0])
                 for k, d in snow]
    dogs_glsblr = [(k, subsample_classes(dog_range, 5, d,
                    indexes=dogs_i)[0]) for k, d in glsblr]

    dogs = [*dogs_sat, *dogs_elst, *dogs_snow, *dogs_glsblr]

    # == reptiles ==
    rep_sat = [(k, subsample_classes(rep_range, 5, d, indexes=reps_i)[0])
               for k, d in sat]
    rep_elst = [(k, subsample_classes(rep_range, 5, d, indexes=reps_i)[0])
                for k, d in elst]
    rep_snow = [(k, subsample_classes(rep_range, 5, d, indexes=reps_i)[0])
                for k, d in snow]
    rep_glsblr = [(k, subsample_classes(rep_range, 5, d, indexes=reps_i)[0])
                  for k, d in glsblr]

    reptiles = [*rep_sat, *rep_elst, *rep_snow, *rep_glsblr]

    # == Generate Images ==
    buffs: List[Tuple[str, str, BytesIO]] = []
    for name, model in tqdm.tqdm(models_list):
        buffs.append((name, 'dogs_base_case',
                     gradcam.gradcam_on_batch_with_target(model, dogs_img)))
        buffs.append((name, 'reptiles_base_case',
                     gradcam.gradcam_on_batch_with_target(model, reptiles_img)))
        for key, reptile in reptiles:
            buff = gradcam.gradcam_on_batch_with_target(model, reptile)
            buffs.append((name, f'reptile_{key}', buff))

        for key, dog in dogs:
            buff = gradcam.gradcam_on_batch_with_target(model, dog)
            buffs.append((name, f'dog_{key}', buff))
    return buffs


def get_n_correct(loader: DataLoader, model: nn.Sequential, n: int, gradcam, transform, device):
    res = None
    for inputs, labels in loader:
        inputs = inputs.to(device)
        output: torch.Tensor = model(inputs).argmax(dim=1)
        output = output.cpu()
        inputs = inputs.cpu()
        
        
        
        if res == None:
            res = (inputs[output == labels], labels[output == labels])
        else:
            old_in, old_lab = res
            new_in = torch.concat((torch.Tensor(old_in), inputs[output == labels]))
            new_lab = torch.concat((old_lab, labels[output == labels]))
            res = (new_in, new_lab)
        if res[0].shape[0] >= n:
            return res
    return res


@GenerateImages("glass_blur_grad")
def generate_glass_blur_grad(args, data, models_list, gradcam, transform):
    imagenet = DataLoader(torchvision.datasets.ImageFolder(
        f'{args.imagenet_path}/val', transform), shuffle=True, batch_size=10)
    dog_range = list(range(151, 275))
    rep_range = list(range(29, 68))
    dogs_img, dogs_i = subsample_classes(dog_range, 5, imagenet)
    reptiles_img, reps_i = subsample_classes(rep_range, 5, imagenet)

    glsblr = [(k, DataLoader(d.dataset, shuffle=True, batch_size=10)) for k, d in data.items(
    ) if k.startswith('blur_glass_blur')]

    dogs_glsblr = [(k, subsample_classes(dog_range, 5, d,
                    indexes=dogs_i)[0]) for k, d in glsblr]
    rep_glsblr = [(k, subsample_classes(rep_range, 5, d, indexes=reps_i)[0])
                  for k, d in glsblr]

    buffs: List[Tuple[str, str, BytesIO]] = []

    for name, model in tqdm.tqdm(models_list, "Glass blur"):
        buffs.append(
            (name, f'dog/dog_base', gradcam.gradcam_on_batch_with_target(model, dogs_img)))
        buffs.append((name, f'reptile/rep_base',
                     gradcam.gradcam_on_batch_with_target(model, reptiles_img)))

        for key, item in dogs_glsblr:
            buffs.append(
                (name, f'/dog/dog_{key}', gradcam.gradcam_on_batch_with_target(model, item)))

        for key, item in rep_glsblr:
            buffs.append(
                (name, f'/reptile/reptile_{key}', gradcam.gradcam_on_batch_with_target(model, item)))

    return buffs


@GenerateImages("randomly_smapled_set_2")
def generate_randomly_smapled_set_2(args, data, models_list, gradcam, transform):
    # -- saturate
    sat = [(k, DataLoader(d.dataset, shuffle=True, batch_size=50)) for k, d in data.items(
    ) if k.startswith('extra_saturate')]

    # -- elastic transform
    elst = [(k, DataLoader(d.dataset, shuffle=True, batch_size=50)) for k, d in data.items(
    ) if k.startswith('digital_elastic_transform')]

    # -- snow
    snow = [(k, DataLoader(d.dataset, shuffle=True, batch_size=50)) for k, d in data.items(
    ) if k.startswith('weather_snow')]

    # -- glass blur
    glsblr = [(k, DataLoader(d.dataset, shuffle=True, batch_size=50)) for k, d in data.items(
    ) if k.startswith('blur_glass_blur')]

    # == Generate Images ==
    buffs: List[Tuple[str, str, BytesIO]] = []
    for name, model in tqdm.tqdm(models_list):
        for key, loader in sat:
            batch = get_n_correct(loader, model, 10)
            buffs.append(
                (name, f"sat_{key}", gradcam.gradcam_on_batch_with_target(model, batch)))

        for key, loader in elst:
            batch = get_n_correct(loader, model, 10)
            buffs.append(
                (name, f"elst_{key}", gradcam.gradcam_on_batch_with_target(model, batch)))

        for key, loader in snow:
            batch = get_n_correct(loader, model, 10)
            buffs.append(
                (name, f"snow_{key}", gradcam.gradcam_on_batch_with_target(model, batch)))

        for key, loader in glsblr:
            batch = get_n_correct(loader, model, 10)
            buffs.append(
                (name, f"glass_{key}", gradcam.gradcam_on_batch_with_target(model, batch)))
    return buffs


@GenerateImages("regression_test")
def generate_regression_test(args, data, models_list, gradcam, transform):
    imagenet = DataLoader(torchvision.datasets.ImageFolder(
        f'{args.imagenet_path}/val', transform), shuffle=True, batch_size=10)
    dog_range = list(range(151, 275))
    rep_range = list(range(29, 68))
    dogs_img, dogs_i = subsample_classes(dog_range, 5, imagenet)
    reptiles_img, reps_i = subsample_classes(rep_range, 5, imagenet)

    random_list = [(k, DataLoader(d.dataset, shuffle=True, batch_size=10)) for k, d in data.items()]

    dogs = [(k, subsample_classes(dog_range, 5, d,
                    indexes=dogs_i)[0]) for k, d in random_list]
    reps = [(k, subsample_classes(rep_range, 5, d, indexes=reps_i)[0])
                  for k, d in random_list]
    
    import random
    dogs = random.choices(dogs,k=10)
    reps = random.choices(reps,k=10)
    
    buffs: List[Tuple[str, str, BytesIO]] = []

    for name, model in tqdm.tqdm(models_list, "Random Choises"):
        buffs.append(
            (name, f'dog/dog_base', gradcam.gradcam_on_batch_with_target(model, dogs_img)))
        buffs.append((name, f'reptile/rep_base',
                     gradcam.gradcam_on_batch_with_target(model, reptiles_img)))

        for key, item in dogs:
            buffs.append(
                (name, f'/dog/dog_{key}', gradcam.gradcam_on_batch_with_target(model, item)))

        for key, item in reps:
            buffs.append(
                (name, f'/reptile/reptile_{key}', gradcam.gradcam_on_batch_with_target(model, item)))

    return buffs

def main(args):
    
    device = torch.device(
    f"cuda{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    
    print("GPU to be used: ", device)

    gradcam = GradcamModule(device)
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    
    
    data = ImagenetCDataloader(
        args.test_dataset, transform,
        shuffle=True,
        num_workers=4,
        batch_size=10
    ).datasets

    barlow = load_model(
        "ssl_robustness/models/barlow/resnet50_linear-8xb32-coslr-100e_in1k_20220825-52fde35f.pth", device, "backbone")

    byol = load_model(
        "ssl_robustness/models/byol/resnet50_linear-8xb512-coslr-90e_in1k_20220825-7596c6f5.pth", device, "backbone")

    simclr = load_model(
        "ssl_robustness/models/simclr/resnet50_linear-8xb512-coslr-90e_in1k_20220825-f12c0457.pth", device, "backbone")

    simsam = load_model(
        "ssl_robustness/models/simsam/resnet50_linear-8xb512-coslr-90e_in1k_20220825-f53ba400.pth", device, "backbone")

    swav = load_model(
        "ssl_robustness/models/swav/resnet50_linear-8xb32-coslr-100e_in1k_20220825-80341e08.pth", device, "backbone")

    dino = load_model("ssl_robustness/models/dino/dino.pth", device)

    models_list = [("barlow", barlow), ("byol", byol),
                   ("simclr", simclr), ("simsam", simsam),
                   ("swav", swav), ("dino", dino)]

    
    # Call Functions
    #generate_regression_test(data, models_list)
    generate_full_samples(args, data, models_list, gradcam, transform)
    generate_randomly_smapled_set(args, data, models_list,gradcam, transform)
    generate_randomly_smapled_set_2(args, data, models_list, gradcam, transform)
    generate_glass_blur_grad(args, data, models_list, gradcam, transform)
    
    

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description='Generate Gradcams for all SSL methods')
    parser.add_argument('--test_dataset', type=str, default='ssl_robustness/data/imagenet_c', help='location of the ImageNet-C dataset in this version')
    parser.add_argument('--imagenet_path', type=str, default='ssl_robustness/data/imagenet', help='location of the ImageNet dataset')
    parser.add_argument('--output', type=str, default='ssl_robustness/output/', help='location of output directory for saving results')
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    
    args = parser.parse_args()
    main(args)
    