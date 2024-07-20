import os
import shutil

T2_SEQ = 0
ADC_SEQ = 1
DWI_SEQ = 2

source_train_dir = os.path.join('Prostate158', 'prostate158_train', 'train')
source_test_dir = os.path.join('Prostate158', 'prostate158_test', 'prostate158_test', 'test')
dest_dir = 'nnUNet_raw/Dataset158_Prostate158'

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

    if os.path.exists(t2_path):
        shutil.copy(t2_path, os.path.join(images_dir, get_image_filename(case_number, T2_SEQ)))
    else:
        raise FileNotFoundError(f'No T2w file found for case {case_number}!')

    if os.path.exists(adc_path):
        shutil.copy(adc_path, os.path.join(images_dir, get_image_filename(case_number, ADC_SEQ)))
    else:
        raise FileNotFoundError(f'No ADC file found for case {case_number}!')

    if os.path.exists(dwi_path):
        shutil.copy(dwi_path, os.path.join(images_dir, get_image_filename(case_number, DWI_SEQ)))
    else:
        raise FileNotFoundError(f'No DWI file found for case {case_number}!')

    if os.path.exists(adc_tumor_reader1_path):
        shutil.copy(adc_tumor_reader1_path, os.path.join(labels_dir, get_label_filename(case_number)))
    elif os.path.exists(adc_tumor_reader_1_empty_path):
        shutil.copy(adc_tumor_reader_1_empty_path, os.path.join(labels_dir, get_label_filename(case_number)))
    else:
        raise FileNotFoundError(f'No ADC Tumor Reader file found for case {case_number}!')

def process_directory(source_dir, images_dir, labels_dir, start_case_number=0):
    files = [f for f in os.listdir(source_dir) if not f.startswith('.')]
    for case_number, case_dir in enumerate(sorted(files), start=start_case_number):
        case_path = os.path.join(source_dir, case_dir)
        if os.path.isdir(case_path):
            process_case(case_path, case_number, images_dir, labels_dir)
        else:
            raise NotADirectoryError(f'{case_path} is not a directory!')

process_directory(source_train_dir, imagesTr_dir, labelsTr_dir, start_case_number=19)
print("Training files copied and renamed successfully.")

process_directory(source_test_dir, imagesTs_dir, labelsTs_dir)
print("Test files copied and renamed successfully.")