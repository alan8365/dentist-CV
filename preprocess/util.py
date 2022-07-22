import os
import json

import pandas
import pandas as pd

from glob import glob


def get_image_label(save=True):
    dataset_dir = os.path.join('..', '..', 'Datasets', 'phase-2')
    json_files = glob(os.path.join(dataset_dir, '**', '*.json'), recursive=True)

    col_names = ['13', '17', '23', '27', '33', '37', '43', '47',
                 'Imp', 'R.R', 'bridge', 'caries', 'crown', 'embedded',
                 'endo', 'filling', 'impacted', 'post']

    d = {}
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

            shapes = data['shapes']
            labels = {shape['label'] for shape in shapes}

            filename, _ = os.path.splitext(os.path.basename(json_file))

            d.update({filename: {i: (i in labels) for i in col_names}})

    df = pd.DataFrame.from_dict(d, orient='index')
    if save:
        df.to_csv('label_TF.csv', index=True, index_label='filename')

    return df


def get_image_by_labels(target_labels):
    df = pd.read_csv('label_TF.csv', index_col='filename')

    result_mask = df[target_labels].any(axis=1)

    return df[result_mask]


# TODO classified by tooth number
if __name__ == '__main__':
    a = get_image_by_labels(['caries', 'crown', 'endo', 'post'])
