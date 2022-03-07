import copy

import mmcv
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.parallel import MMDataParallel

from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from mmcls.apis import inference_model, init_model, show_result_pyplot
import cv2
import numpy as np
import torch
from mmcv.parallel import collate, scatter
import torch.nn as nn
import torchvision
import os
from PIL import Image
from matplotlib import pyplot as plt
from mmcls.datasets.pipelines import Compose
from torchvision import transforms as pth_transforms
from PIL import Image, ImageDraw
from skimage.measure import find_contours
from matplotlib.patches import Polygon


company_colors = [
    (0,160,215), # blue
    (220,55,60), # red
    (245,180,0), # yellow
    (10,120,190), # navy
    (40,150,100), # green
    (135,75,145), # purple
]
company_colors = [(float(c[0]) / 255.0, float(c[1]) / 255.0, float(c[2]) / 255.0) for c in company_colors]



class FeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.inputs = []
        self.features = []

    def hook_fn(self, module, input, output):
        x = copy.deepcopy(input)[0].cpu().numpy()
        out = copy.deepcopy(output)[0].cpu().numpy()
        self.inputs.append(x)
        self.features.append(out)

    def remove(self):
        self.hook.remove

def apply_mask2(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    t= 0.2
    mi = np.min(mask)
    ma = np.max(mask)
    eps = 1e-6
    mask = (mask - mi) / (ma - mi + eps)
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * np.sqrt(mask) * (mask>t)) + alpha * np.sqrt(mask) * (mask>t) * color[c] * 255
    return image

def show_attn(img, attentions):
    w_featmap = 224 // 16
    h_featmap = 224 // 16
    nh = attentions.shape[0]

    # 排序后的值与坐标索引
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - 0.6)
    idx2 = torch.argsort(idx)
    # idx2 表示第idx个现在th_attn的idx2[idx]的位置
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    # interpolate
    th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, h_featmap, w_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()
    os.makedirs('pics/', exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True),
                                 os.path.join('pics/', "img" + ".png"))

    img = Image.open(os.path.join('pics/', "img" + ".png"))
    margin = 15
    num = nh // 2
    attns = Image.new('RGB', (attentions.shape[2] * num + (num - 1) * margin, attentions.shape[1]), (255, 255, 255))
    for j in range(nh // 2):
        fname = os.path.join('pics/', "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        attns.paste(Image.open(fname), (j * (attentions.shape[2] + margin), 0))

    return attentions, th_attn, img, attns

def show_attn_color(image, attentions, th_attn, index=None, head=[6, 7, 8, 9, 10, 11]):
    M = image.max()
    m = image.min()
    span = 64
    image = ((image - m) / (M - m)) * span + (256 - span)
    image = image.mean(axis=2)
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

    for j in head:
        m = attentions[j]
        m *= th_attn[j]
        attentions[j] = m
    mask = np.stack([attentions[j] for j in head])

    blur = False
    contour = False
    alpha = 1
    figsize = tuple([i / 100 for i in [224, 224]])
    fig = plt.figure(figsize=figsize, frameon=False, dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    if len(mask.shape) == 3:
        N = mask.shape[0]
    else:
        N = 1
        mask = mask[None, :, :]

    # AJ
    for i in range(N):
        mask[i] = mask[i] * (mask[i] == np.amax(mask, axis=0))
    a = np.cumsum(mask, axis=0)
    for i in range(N):
        mask[i] = mask[i] * (mask[i] == a[i])

    colors = company_colors[:N]

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = 0.1 * image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask, (10, 10))
        # Mask
        masked_image = apply_mask2(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros(
                (_mask.shape[0] + 2, _mask.shape[1] + 2))  # , dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    ax.axis('image')
    # fname = os.path.join(output_dir, 'bnw-{:04d}'.format(imid))
    prefix = f'id{index}_' if index is not None else ''
    fname = os.path.join('pics/', "attn_color.png")
    fig.savefig(fname)
    attn_color = Image.open(fname)
    return attn_color


def main():
    CLASSES = ('BAS', 'EBO', 'EOS', 'LYT', 'MON', 'MYB', 'MYO', 'NGB', 'NGS', 'PMO')
    config = 'configs/vision_transformer/vit-base-cell-p16_pt-64xb64_in1k-224.py'
    checkpoint = 'work_dir/vision_transformer_cell_test/epoch_52.pth'
    imgs = ['data/imagenet/train/EOS/EOS_0055.jpg', 'data/imagenet/val/BAS_0025_aug_2.jpg',
            'data/imagenet/val/EBO_0005_aug_0.jpg', 'data/imagenet/val/EOS_0268.jpg', 'data/imagenet/val/NGS_5592.jpg',
            'data/imagenet/val/MYO_1094.jpg', 'data/imagenet/val/EOS_0129.jpg', 'data/imagenet/val/BAS_0040.jpg']
    # path = 'data/imagenet/val/'
    # imgs = [os.path.join(path, img_file) for img_file in os.listdir(path)]
    model = init_model(config, checkpoint, device='cuda:0')
    sparse_attn = model.backbone.attn_select
    attn_hook = FeatureHook(sparse_attn)
    for img_file in imgs:
        result = inference_model(model, img_file)
    # attns = attn_hook.inputs
    # attentions = []
    # for attn in attns:
    #     print(attn.shape)
    #     attentions.append(attn[-2, 0, :, 0, 1:])
    attns = attn_hook.features
    attentions = []
    for attn in attns:
        attentions.append(attn)
    mmcv.dump(attentions, 'tmp.pkl')


if __name__ == '__main__':
    main()
    attns = mmcv.load('tmp.pkl')
    imgs = ['data/imagenet/train/EOS/EOS_0055.jpg', 'data/imagenet/val/BAS_0025_aug_2.jpg',
            'data/imagenet/val/EBO_0005_aug_0.jpg', 'data/imagenet/val/EOS_0268.jpg', 'data/imagenet/val/NGS_5592.jpg',
            'data/imagenet/val/MYO_1094.jpg', 'data/imagenet/val/EOS_0129.jpg', 'data/imagenet/val/BAS_0040.jpg']
    # path = 'data/imagenet/val/'
    # imgs = [os.path.join(path, img_file) for img_file in os.listdir(path)]
    transform = pth_transforms.Compose([
        pth_transforms.Resize([224, 224]),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    n = len(imgs)
    m = 224
    h, w = 14, 14
    margin = 15

    # for i, img_file in enumerate(imgs):
    #     img = Image.open(img_file)
    #     img = transform(img)
    #     attentions, th_attn, pic_i, pic_attn = show_attn(img, torch.from_numpy(attns[i]))
    #     pic_attn_color = show_attn_color(img.permute(1, 2, 0).cpu().numpy(), attentions, th_attn)
    #     fig = Image.new('RGB', (224, 224))
    #     fig.paste(pic_attn_color, (0, 0))
    #     base_file = os.path.basename(img_file).split('.')[0]
    #     fig.save(os.path.join('./pics', base_file + '_color.jpg'))

    final_pic = Image.new('RGB', (m * 8 + 7 * margin, m * n + (n - 1) * margin),  "#FFFFFF")
    for i, img_file in enumerate(imgs):
        img = Image.open(img_file)
        img = transform(img)
        # make the image divisible by the patch size
        attn = torch.from_numpy(attns[i])
        _, patch_select = attn.max(1)
        attentions, th_attn, pic_i, pic_attn = show_attn(img, attn)
        pic_attn_color = show_attn_color(img.permute(1, 2, 0).cpu().numpy(), attentions, th_attn)
        final_pic.paste(pic_i, (0, i * (m + margin)))
        final_pic.paste(pic_attn_color, (pic_i.size[1] + margin, i * (m + margin)))
        final_pic.paste(pic_attn, ((pic_i.size[1] + margin) * 2, i * (m + margin)))
    final_pic.save('./pics/final.png')

    pic = Image.new('RGB', (m * n + (n - 1) * margin, m * 2 + margin), "#FFFFFF")
    for i, img_file in enumerate(imgs):
        img = Image.open(img_file)
        img_t = transform(img)
        img = img.resize((224, 224))
        draw = ImageDraw.Draw(img)
        attn = torch.from_numpy(attns[i])
        attentions, th_attn, pic_i, pic_attn = show_attn(img_t, attn)
        pic_attn_color = show_attn_color(img_t.permute(1, 2, 0).cpu().numpy(), attentions, th_attn)
        pic.paste(pic_attn_color, (i * (m + margin), m + margin))
        _, patch_select = attn.max(1)
        for idx in patch_select:
            idx = idx.item()
            if idx > 150:
                continue
            row, col = idx // w, idx % w
            x1, y1 = row * h, col * w
            x2, y2 = (row + 1) * h, (col + 1) * w
            draw.rectangle(xy=(x1, y1, x2, y2), fill=None, outline='red', width=3)
            # 计算左上与右下的坐标
        pic.paste(img, (i * (m + margin), 0))
        print(patch_select)
    pic.save('./pics/patch_select.png')

    # for i, img_file in enumerate(imgs):
    #     img = Image.open(img_file)
    #     img = img.resize((224, 224))
    #     draw = ImageDraw.Draw(img)
    #     attn = torch.from_numpy(attns[i])
    #     _, patch_select = attn.max(1)
    #     for idx in patch_select:
    #         idx = idx.item()
    #         row, col = idx // w, idx % w
    #         x1, y1 = row * h, col * w
    #         x2, y2 = (row + 1) * h, (col + 1) * w
    #         draw.rectangle(xy=(x1, y1, x2, y2), fill=None, outline='red', width=3)
    #         # 计算左上与右下的坐标
    #     fig = Image.new('RGB', (224, 224))
    #     fig.paste(img, (0, 0))
    #     base_file = os.path.basename(img_file).split('.')[0]
    #     fig.save(os.path.join('./pics', base_file + '_select.jpg'))
    #     print(patch_select)

    # attn_map with the shape [num_layer, B, num_heads, N, N]
