from analysis.attention_map.visualize_attention import company_colors, apply_mask2

transform = pth_transforms.Compose([
    pth_transforms.Resize([480, 480]),
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def show_attn(img, index=None):
    w_featmap = img.shape[-2] // 16
    h_featmap = img.shape[-1] // 16

    attentions = vit.get_last_selfattention(img.cuda())

    nh = attentions.shape[1]  # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - 0.6)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    # interpolate
    th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()

    # save attentions heatmaps
    prefix = f'id{index}_' if index is not None else ''
    os.makedirs('pics/', exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True),
                                 os.path.join('pics/', "img" + ".png"))
    img = Image.open(os.path.join('pics/', "img" + ".png"))

    attns = Image.new('RGB', (attentions.shape[2] * nh, attentions.shape[1]))
    for j in range(nh):
        fname = os.path.join('pics/', "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        attns.paste(Image.open(fname), (j * attentions.shape[2], 0))

    return attentions, th_attn, img, attns


def show_attn_color(image, attentions, th_attn, index=None, head=[0, 1, 2, 3, 4, 5]):
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
    figsize = tuple([i / 100 for i in [480, 480]])
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


img = transform(img)
# make the image divisible by the patch size
w, h = img.shape[1] - img.shape[1] % 16, img.shape[2] - img.shape[2] % 16
img = img[:, :w, :h].unsqueeze(0)
attentions, th_attn, pic_i, pic_attn = show_attn(img)
pic_attn_color = show_attn_color(img[0].permute(1, 2, 0).cpu().numpy(), attentions, th_attn)
final_pic = Image.new('RGB', (pic_i.size[1] * 2 + pic_attn.size[0], pic_i.size[1]))
final_pic.paste(pic_i, (0, 0))
final_pic.paste(pic_attn_color, (pic_i.size[1], 0))
final_pic.paste(pic_attn, (pic_i.size[1] * 2, 0))
display(final_pic)