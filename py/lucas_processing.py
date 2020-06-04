#!/usr/bin/env python
# coding: utf-8
"""Processing functions for the LUCAS dataset."""

import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import lucas_utils as utils


def dimension_reduction(csv_path: str,
                        path_to_chunks: str = "../data/chunks",
                        chunksize: int = 1000,
                        verbose=0):
    """Reduce dimension of the 0.5 nm resolution spectra.

    To reduce the dimension of the dataset, we aim for 256 spectral channels
    (so multiple of 2) to make calculations easier later on. Therefore, we
    need to mean over 4200 / 256 = 16.4 channels. Since that is not possible,
    we combine 16 channels for the first 152 times and combine then 17
    channels for 104 times because 16 * 152 + 17 * 104 = 4200 (linear system).

    Parameters
    ----------
    csv_path : str
        Path to the CSV file "LUCAS_TOPSOIL_v1_spectral.csv" of the LUCAS
        dataset.
    path_to_chunks : str
        Path to the temporary data chunks
    chunksize : int
        Size of the chunks
    verbose : int, optional (default=0)
        Controls the verbosity.

    """
    i = 0
    for chunk in pd.read_csv(csv_path, sep=r'/t', engine="python",
                             chunksize=chunksize):
        if verbose:
            print("Chunk: ", i)

        df = chunk

        # get hyperspectral bands and simplify column names
        hypbands_original_str_extended = [
            col for col in df.columns if utils.is_number(col[1:-1])]
        hypbands_original_str = [
            col[1:-1] for col in hypbands_original_str_extended]
        hypbands_original_float = [
            float(col) for col in hypbands_original_str]
        rename_dict = {key: value for (key, value) in zip(
            hypbands_original_str_extended, hypbands_original_float)}
        df.rename(columns=rename_dict, inplace=True)

        # get new hyperspectral bands
        hypbands_left = hypbands_original_float[0:16*152:16]
        hypbands_right = hypbands_original_float[16*152::17]
        hypbands_new = hypbands_left + hypbands_right

        if verbose > 1:
            print("Creating new file and mean over 16 and 17 columns ...")
        df_new = df.drop(hypbands_original_float, axis=1)
        for band in hypbands_new:
            if band in hypbands_left:
                df_new[band] = df[[float(band) + i*0.5
                                   for i in range(16)]].mean(axis=1)
            elif band in hypbands_right:
                df_new[band] = df[[float(band) + i*0.5
                                   for i in range(17)]].mean(axis=1)
            else:
                print("Error: wrong hyperspectral band {0}!".format(band))

        if verbose > 1:
            print("Done!")
            print("Export to new CSV ...")
            df_new.to_csv(
                path_to_chunks+"lucas_newcsv_full_10nm_"+str(i)+".csv")
            print("Done!")
        i += 1


def concat_chunks(path_to_chunks: str,
                  output_file: str = "../data/2_lucas_fromchunks_combined.csv",
                  verbose=0):
    """Concatenate data chunks to one file.

    Parameters
    ----------
    path_to_chunks : str
        Path to the temporary data chunks
    output_file : str, optional
        Path to output csv file
    verbose : int, optional (default=0)
        Controls the verbosity.

    """
    path_to_chunks = os.path.join(path_to_chunks, '')

    # files that contain valuable information
    paths = ['lucas_newcsv_full_10nm_?.csv',
             'lucas_newcsv_full_10nm_1?.csv',
             'lucas_newcsv_full_10nm_2?.csv',
             'lucas_newcsv_full_10nm_3[0-5].csv']

    df_list = []
    for file_path in paths:
        for f in glob.glob(path_to_chunks+file_path):
            if verbose:
                print(f)
            if os.path.isfile(f):
                df_temp = pd.read_csv(f, index_col=0)
                df_list.append(df_temp)

    pd.concat(df_list, axis=0).to_csv(output_file)


def combine_hyp_samples(hyp_csv_path: str, output_path: str, verbose=0):
    """Combine hyperspectral samples.

    Parameters
    ----------
    hyp_csv_path : str
        Path to output csv file from `concat_chunks()`
    output_path : str
        Path to output csv file
    verbose : int, optional (default=0)
        Controls the verbosity.

    """
    df_hyp = pd.read_csv(hyp_csv_path, index_col=0)
    df_hyp.rename(columns={'"ID"': "ID"}, inplace=True)

    def simplify_str(s):
        if isinstance(s, str):
            return s[1:-1]
        return s

    df_hyp.ID = df_hyp.ID.apply(simplify_str)

    id_min = 1  # 18
    id_max = 81000  # 80093

    def get_poss_id(my_id):
        return [str(my_id), str(my_id)+"-1", str(my_id)+"-2",
                str(my_id)+"UK", str(my_id)+"UK-1", str(my_id)+"UK-2"]

    df_hyp_new_list = []
    for curr_id in tqdm(range(id_min, id_max), desc="IDs"):
        id_list = []

        for poss_id in get_poss_id(curr_id):
            if poss_id in df_hyp.ID.values:
                id_list.append(poss_id)

        if id_list != []:
            df_hyp_new_list.append(
                pd.DataFrame(df_hyp.loc[df_hyp["ID"] == id_list[0]]))
    df_hyp_new = pd.concat(df_hyp_new_list, axis=0)

    def simplify_id(s):
        if str(s)[-4:-1] == "UK-":
            return s[:-4]
        if str(s)[-2] == "-":
            return s[:-2]
        if str(s)[-2:] == "UK":
            return s[:-2]
        return s

    df_hyp_new.ID = df_hyp_new.ID.apply(simplify_id)
    if verbose:
        print("New shape:", df_hyp_new.shape)
    df_hyp_new.to_csv(output_path)


def combine_csv_xls(hyp_csv_path: str,
                    other_csv_path: str = "../data/LUCAS_TOPSOIL_v1.csv",
                    output_path: str = "../data/lucas_new_csv_full_10nm.csv",
                    verbose=0):
    """Combine CSV and XLS file.

    Parameters
    ----------
    hyp_csv_path : str
        Path to output csv file from `combine_hyp_samples()`
    other_csv_path : str, optional
        Path to the LUCAS file `LUCAS_TOPSOIL_v1.csv`.
    output_path : str, optional
        Path to output csv file
    verbose : int, optional (default=0)
        Controls the verbosity.

    """
    if verbose:
        print("Start combining csv files ...")
    df_hyp = pd.read_csv(hyp_csv_path)
    df_data = pd.read_csv(other_csv_path, sep=";")

    if verbose:
        print("Shape of hyp-file:", df_hyp.shape)
        print("Shape of data-file:", df_data.shape)
    df_hyp.ID = df_hyp.ID.apply(pd.to_numeric)

    hypbands = [col for col in df_hyp.columns if utils.is_number(col)]
    sampleids = df_data.sample_ID.values

    # 1. find inconsistent sampleIDs -> drop
    notinhyp = [id for id in sampleids if id not in df_hyp.ID.values]
    notindat = [id for id in df_hyp.ID.values if id not in sampleids]
    if verbose:
        print("notinhyp", len(notinhyp))
        print("notindat", len(notindat))
    df_hyp = df_hyp[~df_hyp["ID"].isin(notindat)]
    df_data = df_data[~df_data["sample_ID"].isin(notinhyp)]
    if verbose:
        print("Dropped inconsistent sampleIDs. Shapes: {0}, {1}".format(
            df_hyp.shape, df_data.shape))

    # 2. merge both dataframes -> nearly 20 000 datapoints
    if verbose:
        print("- Merging ...")
    data_full = pd.merge(left=df_data, right=df_hyp, how="inner",
                         left_on="sample_ID", right_on="ID", sort=True)

    # 3. drop nan
    if verbose:
        print("- Dropping nan ...")
        print("  Before: {0}".format(data_full.shape))
    data_full.dropna(subset=hypbands+["clay", "sand", "silt"], inplace=True,
                     axis=0)
    if verbose:
        print("- After: {0}".format(data_full.shape))

    # 4. save to file
    if verbose:
        print("- Saving csv file with shape {0} ...".format(data_full.shape))
    data_full.to_csv(output_path)
    if verbose:
        print("Done!")


def add_categories_and_superclasses(input_path: str,
                                    output_path: str,
                                    verbose=0):
    """Add categories and superclasses to dataframes.

    Parameters
    ----------
    input_path : str
        Path to output csv file of `combine_csv_xls()`
    output_path : str
        Path to output csv file
    verbose : int, optional (default=0)
        Controls the verbosity.

    """
    if verbose:
        print("Start adding categories and superclasses to files ...")

    df_full = pd.read_csv(input_path)

    # drop rows with clay+sand+silt = 0 or close
    if verbose:
        print("- Dropping clay+sand+silt < 0.98 ...")
    df_full = df_full[df_full[["clay", "sand", "silt"]].sum(axis=1) >= 98.]

    # add "category"
    if verbose:
        print("- Adding 'category' ...")
    df_full["category"] = np.vectorize(
        utils.soil_texture_classes, otypes=[np.str])(
            df_full["clay"], df_full["sand"], df_full["silt"])

    # # add "category_id"
    # if verbose:
    #     print("- Adding 'category_id' ...")
    # df_full["category_id"] = np.vectorize(
    #     utils.get_soil_texture_id)(df_full["category"])

    # add "superclass" and "superclass_id"
    if verbose:
        print("- Adding 'superclass' ...")
    df_full["superclass"] = np.vectorize(
        utils.get_soil_texture_superclass)(df_full["category"])
    df_full["superclass_id"] = np.vectorize(
        utils.get_soil_texture_superclass_id)(df_full["superclass"])

    # drop dummy columns
    if verbose:
        print("- Dropping dummy columns ...")
    df_full = df_full.loc[:, ~df_full.columns.str.contains('^Unnamed')]

    # saving to file
    if verbose:
        print("- Saving full dataframe ...")
    df_full.to_csv(output_path)
    if verbose:
        print("Done!")


def split_lucas_dataset(full_dataset_path: str = "../data/lucas_new_final.csv",
                        output_path_prefix: str = "../data/lucas_new_final_",
                        random_state=42,
                        train_frac: float = 0.8,
                        val_frac: float = 0.2,
                        verbose=0):
    """Split processed LUCAS dataset.

    Parameters
    ----------
    full_dataset_path : str, optional
        Path to  output csv file of `add_categories_and_superclasses()`
    output_path_prefix : str, optional
        Prefix for output file
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    train_frac : float, optional (default=0.8)
        Training dataset fraction of the full dataset. For example, 0.8 means
        that 100% - 80% = 20% of the full dataset are used for the test.
    val_frac : float, optional (default=0.2)
        Validation dataset fraction of the full dataset. For example, 0.2
        means, that 20% are used for the validation data. This part is taken
        from the training dataset.
    verbose : int, optional (default=0)
        Controls the verbosity.

    Raises
    ------
    ValueError
        Raised if `train_frac`and `val_frac`are too large.

    """
    if verbose:
        print("Splitting files and exporting them ...")

    if val_frac + train_frac > 1.0:
        raise ValueError("Fractions of validation and training too large.")

    df_full = pd.read_csv(full_dataset_path, low_memory=False, index_col=0)

    np.random.seed(random_state)
    msk = np.random.rand(len(df_full)) < train_frac
    train = df_full[msk]
    test = df_full[~msk]

    msk2 = np.random.rand(len(train)) < (val_frac / train_frac)
    validation = train[msk2]
    train = train[~msk2]

    # save split full dataframes
    train.to_csv(output_path_prefix+"train.csv")
    validation.to_csv(output_path_prefix+"validation.csv")
    test.to_csv(output_path_prefix+"test.csv")
    if verbose:
        print("Train subset {0} created (= {1:.1f}%).".format(
            train.shape, 100*(train.shape[0]/df_full.shape[0])))
        print("Validation subset {0} created (= {1:.1f})%.".format(
            validation.shape, 100*(validation.shape[0]/df_full.shape[0])))
        print("Test subset {0} created (= {1:.1f})%.".format(
            test.shape, 100*(test.shape[0]/df_full.shape[0])))

    # save split dataframes with relevant variables
    utils.save_dataframes(path=output_path_prefix, targetvar="superclass_id",
                          output_path="../data/", verbose=verbose)
