import os
import glob
import csv
from PIL import Image

import numpy as np
from sklearn.model_selection import StratifiedKFold

def generate_pathfile_with_kfsplit(data_dir, outdir="./", file_ext="txt", k=5):
    """Generate a text file that lists the data path and label for each rows.
    Look into data directory where inside are class folders containing dataset.
    Each class folder has dataset that belong to one class type.

    Arguments --------------------

        - data_dir (str): path to data directory
        - k (int): how many KFold to create. StratifiedKFold is used to split all dataset 
            in found in the data_dir into K trainset and K testset.

    Output:
        - K number of files: files containing list of absolute path for each data in class folder.
            Each created file will have names with format: 'train_split_i' where 0 <= i <= k-1.

    Example data_dir directory:

    - data_dir/
        |- cat
            |- cat_img0.png
            |- cat_img1.png
            |_ ...
        |- dog
            |- dog_img0.png
            |- dog_img1.png
            |- ...
    """

    # check if a given directory is valid
    if not os.path.isdir(data_dir):
        raise FileNotFoundError("Invalid path: {}. Folder directory containing subfolder (class folder) is not found." 
            "Please check again".format(data_dir))

    delimiter = ' '
    if file_ext == 'txt':
        delimiter = ' '
    elif file_ext == 'csv':
        delimiter = ','
    elif file_ext == 'tsv':
        delimiter = '\t'
    else:
        raise Exception("Cannot output the file as {} extension." 
            "Supported extensions are [txt, csv, tsv]".format(file_ext))

    class_folders = sorted(glob.glob(data_dir))
    data_dirs = [glob.glob(folder_path + '/*') for folder_path in class_folders]

    all_labels = []
    all_dirs = []
    for num_label in range(len(class_folders)):
        labels = [num_label] * len(data_dirs[num_label])
        all_labels.extend(labels)
        all_dirs.extend(data_dirs[num_label])

    skf = StratifiedKFold(n_splits=k)

    X = np.array(all_dirs)
    y = np.array(all_labels)
    for i, (train_ids, test_ids) in enumerate(skf.split(X, y)):
        X_train = X[train_ids]
        y_train = y[train_ids]

        X_test = X[test_ids]
        y_test = y[test_ids]

        train_filename = "train_split_{}.{}".format(i, file_ext)
        train_out_dir = os.path.join(out_dir, filename)
        with open(outdir, 'w') as f:
            for idx in range(len(train_ids)):
                data_dir = X_train[idx]
                label = y_train[idx]
                f.write("{}{}{}".format(data_dir, delimiter, label))
        
        test_filename = "test_split_{}.{}".format(i, file_ext)
        with open(outdir, 'w') as f:
            for idx in range(len(test_ids)):
                data_dir = X_test[idx]
                label = y_test[idx]
                f.write("{}{}{}".format(data_dir, delimiter, label))
            

def get_path_label_pairs(filepath, delimiter=None):
    """Helper function to parse a (.csv, .txt, .tsv) file which
    contains paths to data and it's respective labels
    
    Arguments -------------
        - filepath (str): path to the file.
        - delimiter (str): specify delimiter regardless file extension format.
            if delimiter not specified, will determine from file extension as follows:
                space: .txt
                comma: .csv
                tabs: .tsv 
    
    example row: 
        /path/to/img_0, 0
        /path/to/img_1, 1
        ...
    """
    filename, file_extension = os.path.splitext('/path/to/somefile.ext')
    
    # determine delimiter
    if not delimiter:
        if file_extension == '.txt':
            delimiter = ' '
        elif file_extension == '.csv':
            delimiter = ','
        elif file_extension == '.tsv':
            delimiter = '\t'
        else:
            raise Exception("File {} has unsupported format {} to get path label pairs of the dataset" \
                .format(filepath, file_extension))
    
    with open(filepath) as f:
        path_label_pairs = csv.reader(f, delimiter=delimiter, 
                                          quotechar='"')
    
    image_paths = list(map(lambda pair: str(pair[0]), path_label_pairs))
    labels = list(map(lambda pair: int(pair[1]), path_label_pairs))
    
    # check if each image_paths is valid
    for idx, path in enumerate(image_paths):
        if os.path.isfile(path) == False:
            raise FileNotFoundError("Image path: {} doesn't exists."
                "Check the image path specified in file: {}, line {}".format(path, filepath, idx+1))

    return image_paths, labels
    

    