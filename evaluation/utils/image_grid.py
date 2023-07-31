

from itertools import repeat
import os
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))

    return dst


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


class ImageGrid:
    """Image Grid Class, contains the grid generated from 
    """

    def __init__(self,  dim: Tuple[int, int], size: int = 224, padding: Tuple[int, int, int, int] = (100, 100)) -> None:
        self.padding = padding

        cols, rows = dim

        padding_perm = [1 if i > 0 else 0 for i in padding]

        left, top = padding_perm
        self.dims = (cols + left, rows + top)

        self.cols: List[List[Optional[Image.Image]]] = [
            [None for _ in range(cols + left)] for _ in range(rows + top)]

        if left == 1:
            for i in range(rows+left):
                left = padding[0]
                self.cols[i][0] = Image.fromarray(
                    np.full(shape=(size, left, 3), fill_value=255, dtype=np.uint8))
        if top == 1:
            for i in range(cols+top):
                top = padding[1]
                self.cols[0][i] = Image.fromarray(
                    np.full((top, size, 3), fill_value=255, dtype=np.uint8))

        self.cols[0][0] = Image.fromarray(
            np.full(shape=(top, left, 3), fill_value=255, dtype=np.uint8))

    def get_cell(self, i: int, j: int):
        return self.cols[j][i]

    def edit_cell_asarray(self, i: int, j: int, image: np.array):
        self.cols[j][i] = Image.fromarray(image)

    def edit_cell(self, i: int, j: int, image: Image.Image):
        self.cols[j][i] = image

    def top_title(self, i: int, text: str, fontsize=18):
        img: Image.Image = self.cols[0][i]
        imgd = ImageDraw.Draw(img)

        x = img.width//2 - (len(text) + (fontsize/2))
        y = img.height / 2 + fontsize/2
        font = ImageFont.truetype("./arial.ttf", fontsize)

        imgd.text((x, y), text, fill=(0, 0, 0), font=font)

    def left_title(self, j: int, text: str, fontsize=18):
        img: Image.Image = self.cols[j][0]
        imgd = ImageDraw.Draw(img)


        x = img.width//2 - (len(text) + fontsize)
        y = img.height / 2 + fontsize/2

        font = ImageFont.truetype("./arial.ttf", fontsize)
        imgd.text((x, y), text, fill=(0, 0, 0), font=font)

    def construct_imagegrid(self) -> Image.Image:
        cols, rows = self.dims
        image = None
        assert (sum([0 if item != None else 1 for item in self.cols]) == 0)
        for j in range(rows):
            image_cols = None
            for i in range(cols):

                if image_cols == None:
                    image_cols = self.cols[j][i]
                else:
                    image_cols = get_concat_h(image_cols, self.cols[j][i])

            if image != None:
                image = get_concat_v(image, image_cols)
            else:
                image = image_cols
        return image


def test():
    grid = ImageGrid((2, 2), 224, padding=(100, 100))
    # p p p
    # p r g
    # p b y

    # color = np.uint8(np.ones((100,100))) * 255
    red = np.full((224, 224, 3), 255, np.uint8)

    red[0:224] = (255, 0, 0)

    green = np.uint8(np.zeros((224, 224, 3)))
    green[0:224] = (0, 255, 0)
    blue = np.uint8(np.zeros((224, 224, 3)))
    blue[0:224] = (0, 0, 255)
    yellow = np.full((224, 224, 3), 255, np.uint8)
    yellow[0:224] = (255, 255, 0)

    grid.edit_cell_asarray(1, 1, red)
    grid.edit_cell_asarray(1, 2, green)
    grid.edit_cell_asarray(2, 1, blue)
    grid.edit_cell_asarray(2, 2, yellow)

    grid.top_title(1, "test 1")
    grid.top_title(2, "test 2")

    grid.left_title(1, "test 3")
    grid.left_title(2, "test 4")

    img = grid.construct_imagegrid()
    print("width:", img.width)
    print("height:", img.height)
    img.save("./out/test.png")


def crop_original_image(orginal: Image.Image, offset: int):
    img = orginal.copy()
    length = 224*3
    img = img.crop((length * offset,
                   0, 224 + length*offset, 224))
    return img


def crop_grad_image(orginal: Image.Image, offset: int) -> Image.Image:
    """Crops the gradcam image out, takes the gradcam.
        Can only take from a single row for the moment.

    Args:
        orginal (PIL.Image.Image): source image, generated using using gradcam module
        offset (int): nr on row

    Returns:
        PIL.Image.Image: cropped image