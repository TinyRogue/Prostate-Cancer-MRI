from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
import os

out_dir = os.path.join(os.getcwd(), 'nnUNet_raw', 'Dataset158_Prostate158')
generate_dataset_json(
    out_dir,
    channel_names={
        0: 'T2',
        1: 'ADC',
        2: 'DWI'
    },
    file_ending=".nii.gz",
    labels={
        "background": 0,
        "tumor": 1,
    },
    dataset_name='Dataset158_Prostate158',
    num_training_cases=139
)
