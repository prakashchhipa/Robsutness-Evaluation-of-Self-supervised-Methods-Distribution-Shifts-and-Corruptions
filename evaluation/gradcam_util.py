
from io import BytesIO
import logging
from typing import Tuple
import numpy as np
import cv2
from pytorch_grad_cam import FullGrad, GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def generate_circle(color):
    array = np.zeros((25, 25, 4), np.uint8)
    circle = Image.fromarray(array)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0,0,25,25), outline=(0,0,0), fill=color)
    return circle

red_circle = generate_circle('red')
green_circle = generate_circle('green')
yellow_circle = generate_circle('yellow')

class GradcamModule:
    
    def __init__(self, device) -> None:
         self.device = device
         self.run = 0
         self.report = {}
    
    def generate_dims(self, size, MAX_COL = 8):
        '''
        Genaret a tuple of size parameters that makes a complete shape square based on max_col parameter.
        
        e.g size = 16 with MAX_COL = 8 => (8,8).
        
        The algorithm tries to have as large as allowable column size
        while keeping it in a grid, i.e all rows if filled fully with a colmn
        
        return col, row
        '''

        col = 1
        row = 1
        
        if(size < MAX_COL):
            return (size, 1)
        res= []
        for i in range(1, size+1):
            if size % i == 0:
                row = i
                col = int(size / row)
                if(col <= MAX_COL):
                            res.append((col,row))

        return max(res,key=lambda item:item[0])

    def gradcam(self, model, img):
        # TODO generate colored highlight and label text
        target_layers = [model.layer4]
        input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input_tensor = input_tensor.to(self.device)
        
        with HiddenPrints():
            with FullGrad(model=model, use_cuda=self.device=='cuda', target_layers=target_layers) as cam:
                            grayscale_cams = cam(input_tensor=input_tensor)
                            cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
        cam = np.uint8(255*grayscale_cams[0, :])
        cam = cv2.merge([cam, cam, cam])
        images = np.hstack((np.uint8(255*img), cam , cam_image))
        
        return images
    
    def gradcam_with_target(self, model, img, targets=[]):
        # TODO generate colored highlight and label text
        target_layers = [model.layer4]
        input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input_tensor = input_tensor.to(self.device)
        targets = [ClassifierOutputTarget(i) for i in targets]
        # with HiddenPrints():
        with GradCAM(model=model, use_cuda=self.device=='cuda', target_layers=target_layers) as cam:
                        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
                        cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
            
        cam = np.uint8(255*grayscale_cams[0, :])
        cam = cv2.merge([cam, cam, cam])
        images = np.hstack((np.uint8(255*img), cam , cam_image))
        
        return images
    
    
    def gradcam_on_batch_with_target(self, model: nn.Module, images: torch.Tensor) -> Tuple[str, BytesIO]:
        model.eval()
        logging.debug("running gradcam against targets")
        # for d in data_loader:
        o, l = images
        ts = list(o)
        ls = [tensor.item() for tensor in list(l)]
        col, row = self.generate_dims(len(o), MAX_COL=5)
        rows = []
        
        #report items
        self.run += 1
        actual = [label.item() for label in l]
        o = o.to(self.device)
        outputs = model(o).argmax(dim=1)
        pred =  [output.item() for output in outputs]

        self.report[str(self.run)] = {'actual' : actual, 'pred': pred}
        # print("rows:",row,"columns:",col)
        # for j in tqdm.tqdm(range(row), f"{name}_row"):
        for j in range(row):
            cols = []
            for i in range(col):
                input_tensor= ts[i + j*col]
                target = ls[i+j*col] 
                # pred = model(input_tensor).argmax(dim=1)
                input_tensor = input_tensor.cpu()
                img = input_tensor.permute(1,2,0).numpy()
                img /= np.amax(img)
                img = np.clip(img, 0,1)
                cam_image = self.gradcam_with_target(model=model, img=img, targets=[target])
                cols.append(cam_image)
            rows.append(np.hstack(cols))

        images = np.vstack(rows)
        buffer =  BytesIO()
        image = Image.fromarray(images)

        length = 224 * 3 # lenght of a gradcam
        l = l.to(self.device)
        # _, top_5_out = torch.topk(model(o), 5)
        
        for i,b in enumerate((outputs == l)):
            
            if b:
                image.paste(green_circle,((length - 224) + length*i + (224 - 25), 0),green_circle)
            else:
                # for k in top_5_out:
                    
                image.paste(red_circle,((length - 224) + length*i + (224 - 25), 0), red_circle)
            
        image.save(buffer, format='png')
        return buffer
    
    
    def gradcam_on_batch(self, model: nn.Module, images: torch.Tensor, name='') -> Tuple[str, BytesIO]:
        model.eval()
        logging.debug("running:",name)
        # for d in data_loader:
        o, _ = images
        ts = list(o)
        col, row = self.generate_dims(len(o), MAX_COL=5)
        rows = []
        # print("rows:",row,"columns:",col)
        # for j in tqdm.tqdm(range(row), f"{name}_row"):
        for j in range(row):
            cols = []
            for i in range(col):
                input_tensor= ts[i + j*col]
                img = input_tensor.permute(1,2,0).numpy()
                img /= np.amax(img)
                img = np.clip(img, 0,1)
                cam_image = self.gradcam(model=model, img=img)
                cols.append(cam_image)
            rows.append(np.hstack(cols))

        images = np.vstack(rows)
        buffer =  BytesIO()
        Image.fromarray(images).save(buffer, format='png')
        return name, buffer

    def save_report(self):
        import json
        with open("./out/report_gradcam.json", '+w') as fp:
            json.dump(self.report, fp)