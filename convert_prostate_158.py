import os
import datetime
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from preprocessing import preprocess


T2_SEQ = 0
ADC_SEQ = 1
DWI_SEQ = 2

source_train_dir = os.path.join('..', 'Prostate158', 'prostate158_train', 'train')
source_test_dir = os.path.join('..', 'Prostate158', 'prostate158_test', 'prostate158_test', 'test')
dest_dir = os.path.join('..', 'nnUNet_raw', 'Dataset158_Prostate158')

train_csv = os.path.join(source_train_dir, '..', 'train.csv')
train_valid_csv = os.path.join(source_train_dir, '..', 'valid.csv')
test_csv = os.path.join(source_test_dir, '..', 'test.csv')

train_df = pd.read_csv(train_csv)
valid_df = pd.read_csv(train_valid_csv)
test_df = pd.read_csv(test_csv)

imagesTr_dir = os.path.join(dest_dir, 'imagesTr')
labelsTr_dir = os.path.join(dest_dir, 'labelsTr')
imagesTs_dir = os.path.join(dest_dir, 'imagesTs')
labelsTs_dir = os.path.join(dest_dir, 'labelsTs')

os.makedirs(imagesTr_dir, exist_ok=True)
os.makedirs(labelsTr_dir, exist_ok=True)
os.makedirs(imagesTs_dir, exist_ok=True)
os.makedirs(labelsTs_dir, exist_ok=True)


def get_image_filename(case_number, sequence_number):
    return f'prostate_{case_number:03d}_{sequence_number:04d}.nii.gz'


def get_label_filename(case_number):
    return f'prostate_{case_number:03d}.nii.gz'


def process_case(case_path, case_number, images_dir, labels_dir):
    t2_path = os.path.join(case_path, 't2.nii.gz')
    adc_path = os.path.join(case_path, 'adc.nii.gz')
    dwi_path = os.path.join(case_path, 'dwi.nii.gz')
    adc_tumor_reader1_path = os.path.join(case_path, 'adc_tumor_reader1.nii.gz')
    adc_tumor_reader_1_empty_path = os.path.join(case_path, 'empty.nii.gz')

    t2_out_path = os.path.join(images_dir, get_image_filename(case_number, T2_SEQ))
    adc_out_path = os.path.join(images_dir, get_image_filename(case_number, ADC_SEQ))
    dwi_out_path = os.path.join(images_dir, get_image_filename(case_number, DWI_SEQ))
    adc_tumor_reader1_out_path = os.path.join(labels_dir, get_label_filename(case_number))

    if not os.path.exists(t2_path):
        raise FileNotFoundError(f'No T2w file found for case {case_number}!')

    reference = sitk.ReadImage(t2_path)

    preprocess(t2_path, t2_out_path)

    if os.path.exists(adc_path):
        preprocess(adc_path, adc_out_path, reference=reference)
    else:
        raise FileNotFoundError(f'No ADC file found for case {case_number}!')

    if os.path.exists(dwi_path):
        preprocess(dwi_path, dwi_out_path, reference=reference)
    else:
        raise FileNotFoundError(f'No DWI file found for case {case_number}!')

    if os.path.exists(adc_tumor_reader1_path):
        preprocess(adc_tumor_reader1_path, adc_tumor_reader1_out_path, reference=reference, mode=sitk.sitkNearestNeighbor)
    elif os.path.exists(adc_tumor_reader_1_empty_path):
        preprocess(adc_tumor_reader_1_empty_path, adc_tumor_reader1_out_path, reference=reference, mode=sitk.sitkNearestNeighbor)
    else:
        raise FileNotFoundError(f'No ADC Tumor Reader file found for case {case_number}!')


def process_df(df, src_dir, images_dir, labels_dir, start_case=1):
    with tqdm(total=len(df)) as bar:
        for row_idx, row in df[['ID', 't2', 'adc', 'dwi', 'adc_tumor_reader1']].iterrows():
            case_path = os.path.join(src_dir, str(row['ID']).zfill(3))
            if os.path.isdir(case_path):
                process_case(case_path, row_idx + start_case, images_dir, labels_dir)
                bar.update(1)
            else:
                raise NotADirectoryError(f'{case_path} is not a directory!')


if __name__ == "__main__":
    process_df(test_df, source_test_dir, imagesTs_dir, labelsTs_dir)
    process_df(valid_df, source_train_dir, imagesTr_dir, labelsTr_dir)
    process_df(train_df, source_train_dir, imagesTr_dir, labelsTr_dir, start_case=21)
    generate_dataset_json(
        dest_dir,
        channel_names={
            T2_SEQ: 'T2',
            ADC_SEQ: 'ADC',
            DWI_SEQ: 'DWI'
        },
        file_ending=".nii",
        labels={
            "background": 0,
            "tumor": 1,
        },
        dataset_name=f'Dataset158_Prostate158_{datetime.datetime.now()}',
        num_training_cases=139,
    )
