import matplotlib.pyplot as plt
import SimpleITK as sitk
from center_crop import crop
import os
import numpy as np


def preprocess(img_path, out_path, mode=sitk.sitkLinear, reference=None):
    target = sitk.ReadImage(img_path)
    if reference is not None:
        target = sitk.Resample(target, referenceImage=reference, interpolator=mode)
    target = crop(target, (.15, .15, .0), mode)
    sitk.WriteImage(target, out_path)


def process_slice(data, selected_image, mask=False):
    img_slice = data[selected_image, :, :]
    if mask:
        img_slice = np.ma.masked_where(img_slice == 0, img_slice)
    return img_slice


if __name__ == '__main__':
    f_names = ('t2', 'adc', 'dwi', 'adc_tumor_reader1')
    ext = '.nii.gz'
    files = [os.path.join('001', f'{key}{ext}') for key in f_names]
    images = {f_names[i]: sitk.ReadImage(f) for i, f in enumerate(files)}
    fig, axs = plt.subplots(4, 4)
    selected_slice = 8

    to_show = [process_slice(sitk.GetArrayFromImage(images[f_name]), selected_slice) for f_name in f_names]

    for i, handler in enumerate(to_show):
        axs.flat[i].imshow(handler, cmap='gray')
        axs.flat[i].set_title(f_names[i])

    resampled_images = {key: sitk.Resample(images[key], referenceImage=images['t2']) for key in f_names[:3]}
    resampled_images['adc_tumor_reader1'] = sitk.Resample(images['adc_tumor_reader1'], referenceImage=images['t2'], interpolator=sitk.sitkNearestNeighbor)
    to_show = [process_slice(sitk.GetArrayFromImage(resampled_images[f_name]), selected_slice) for f_name in f_names]
    for i, handler in enumerate(to_show):
        axs.flat[i + len(f_names)].imshow(handler, cmap='gray')

    cropped_images = {key: crop(resampled_images[key], (.15, .15, .0)) for key in f_names}
    to_show = [process_slice(sitk.GetArrayFromImage(cropped_images[f_name]), selected_slice) for f_name in f_names]
    for i, handler in enumerate(to_show):
        axs.flat[i + len(f_names) * 2].imshow(handler, cmap='gray')

    os.makedirs('./test', exist_ok=True)
    for i, f in enumerate(files):
        preprocess(f, f'./test/{f_names[i]}{ext}', mode=sitk.sitkLinear if i == len(files) - 1 else sitk.sitkNearestNeighbor, reference=images['t2'])
    files = [os.path.join('test', f'{key}{ext}') for key in f_names]
    images = {f_names[i]: sitk.ReadImage(f) for i, f in enumerate(files)}
    to_show = [process_slice(sitk.GetArrayFromImage(images[f_name]), selected_slice) for f_name in f_names]
    for i, handler in enumerate(to_show):
        axs.flat[i + len(f_names) * 3].imshow(handler, cmap='gray')

    fig.suptitle('Image preprocessing')
    plt.tight_layout()
    plt.show()
