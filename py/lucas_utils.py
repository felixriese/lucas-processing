"""Utilties for LUCAS dataset."""

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def is_number(s):
    """Check if `s` is a number.

    Parameters
    ----------
    s : str, int, float, list
        Variable for which the function checks, if it is a number.

    Returns
    -------
    bool
        True, if the variable is a number. For example 1, 1.0.
        False, if the variable is not a pure number. For example "a", "1a".

    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_dataframes(path: str):
    """Get dataframes from processed files.

    Parameters
    ----------
    path : str
        Path to CSV files for train, validation and test data.

    Returns
    -------
    DataFrames
        Dataframes for train, validation, and test.

    """
    # load data
    df_train = pd.read_csv(path+"train.csv", index_col=0,
                           delimiter=",", low_memory=False)
    df_validation = pd.read_csv(path+"validation.csv", index_col=0,
                                delimiter=",", low_memory=False)
    df_test = pd.read_csv(path+"test.csv", index_col=0,
                          delimiter=",", low_memory=False)

    # drop nan values
    hypbands = [col for col in df_train.columns if is_number(col)]
    subset = hypbands + ["sand", "clay", "silt"]
    df_train.dropna(how="any", inplace=True, subset=subset)
    df_validation.dropna(how="any", inplace=True, subset=subset)
    df_test.dropna(how="any", inplace=True, subset=subset)

    # shuffle data
    df_train.dropna(how="any", inplace=True, subset=subset)
    df_validation.dropna(how="any", inplace=True, subset=subset)
    df_test = df_test.sample(frac=1, random_state=44)

    return df_train, df_validation, df_test


def get_data(targetvar: str,
             path: str = "../data/4_lucas_new_final_",
             scaling_x: bool = False,
             scaling_y: bool = False):
    """Get processed data.

    Parameters
    ----------
    targetvar : str
        Name of the target variable
    path : str, optional
        Path to the files.
    scaling_x : bool, optional (default=False)
        If true, the x data is scaled.
    scaling_y : bool, optional (default=False)
        If true, the y data is scaled.

    Returns
    -------
    X_train : np.array
        Hyperspectral training data
    X_val : np.array
        Hyperspectral validation data
    X_test : np.array
        Hyperspectral test data
    y_train : np.array
        Training labels
    y_val : np.array
        Validation labels
    y_test : np.array
        Test labels
    hypbands : list
        List of hyperspecral bands
    targetvar : str
        Name of the target variable

    """
    df_train, df_validation, df_test = get_dataframes(path=path)

    return process_dataframes(df_train, df_validation, df_test, targetvar,
                              scaling_x, scaling_y)


def process_dataframes(df_train: pd.DataFrame,
                       df_validation: pd.DataFrame,
                       df_test: pd.DataFrame,
                       targetvar: str,
                       scaling_x: bool = False,
                       scaling_y: bool = False):
    """Process dataframes.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data
    df_validation : pd.DataFrame
        Validation data
    df_test : pd.DataFrame
        Test data
    targetvar : str
        Name of the target variable
    scaling_x : bool, optional (default=False)
        If true, the x data is scaled.
    scaling_y : bool, optional (default=False)
        If true, the y data is scaled.

    Returns
    -------
    X_train : np.array
        Hyperspectral training data
    X_val : np.array
        Hyperspectral validation data
    X_test : np.array
        Hyperspectral test data
    y_train : np.array
        Training labels
    y_val : np.array
        Validation labels
    y_test : np.array
        Test labels
    hypbands : list
        List of hyperspecral bands
    targetvar : str
        Name of the target variable

    """
    # define features and regression target variable
    hypbands = [col for col in df_train.columns if is_number(col)]

    # generate matrices
    # X_train = np.expand_dims(df_train[hypbands].values, axis=2)
    X_train = df_train[hypbands].values
    y_train = df_train[targetvar].values
    X_val = df_validation[hypbands].values
    y_val = df_validation[targetvar].values
    X_test = df_test[hypbands].values
    y_test = df_test[targetvar].values

    if scaling_x:
        # scale input variables
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0],
                                                       X_train.shape[1]))
        X_val = scaler.fit_transform(X_val.reshape(X_val.shape[0],
                                                   X_val.shape[1]))
        X_test = scaler.fit_transform(X_test.reshape(X_test.shape[0],
                                                     X_test.shape[1]))
    if scaling_y:
        # scale target variables
        scaler = MinMaxScaler(feature_range=(0, 1))
        y_train = scaler.fit_transform(y_train.reshape(-1, 1))
        y_val = scaler.fit_transform(y_val.reshape(-1, 1))
        y_test = scaler.fit_transform(y_test.reshape(-1, 1))

    return X_train, X_val, X_test, y_train, y_val, y_test, hypbands, targetvar


def save_dataframes(path: str,
                    targetvar: str,
                    output_path: str = "../data/",
                    verbose=0):
    """Save dataframes to CSV.

    Parameters
    ----------
    path : str
        Path to the dataframes
    targetvar : str
        Name of the target variable
    output_path : str, optional (default="../data/")
        Path to output folder
    verbose : int, optional (default=0)
        Controls the verbosity.

    """
    df_train, df_validation, df_test = get_dataframes(path)

    # define features and regression target variable
    hypbands = [col for col in df_train.columns if is_number(col)]

    df_train.to_csv(output_path+"X_train.csv", columns=hypbands)
    df_train.to_csv(output_path+"y_train.csv", columns=[targetvar])
    df_validation.to_csv(output_path+"X_val.csv", columns=hypbands)
    df_validation.to_csv(output_path+"y_val.csv", columns=[targetvar])
    df_test.to_csv(output_path+"X_test.csv", columns=hypbands)
    df_test.to_csv(output_path+"y_test.csv", columns=[targetvar])

    if verbose:
        print("Data saved!")


def soil_texture_classes(clay, sand, silt):
    """Categories of soil texture.

    Parameters
    ----------
    clay : float
        Percentage of clay between 0...100
    sand : float
        Percentage of sand between 0...100
    silt : float
        Percentage of silt between 0...100

    Returns
    -------
    class_string : str

    Abbreviations
    -------------
        S   sand (German: Sand)
        U   silt (German: Schluff)
        T   clay (German: Ton)

    Source for definitions: http://www.nibis.de/~trianet/soil/boden4.htm

    """
    if abs(sum([clay, sand, silt]) - 100) > 1.:
        return None

    if sum([clay, sand, silt]) < 2.:
        clay *= 100
        sand *= 100
        silt *= 100

    clay = float(clay)
    sand = float(sand)
    silt = float(silt)

    soil_class = ""

    # --- Sand (S)
    if ((0 <= clay <= 5) and (0 <= silt <= 10) and (85 <= sand <= 100)):
        soil_class = "Ss"
    elif ((0 <= clay <= 5) and (10 <= silt <= 25) and (70 <= sand <= 90)):
        soil_class = "Su2"
    elif ((5 <= clay <= 8) and (10 <= silt <= 25) and (67 <= sand <= 85)):
        soil_class = "Sl2"
    elif ((8 <= clay <= 12) and (10 <= silt <= 40) and (48 <= sand <= 82)):
        soil_class = "Sl3"
    elif ((5 <= clay <= 17) and (0 <= silt <= 10) and (73 <= sand <= 95)):
        soil_class = "St2"
    elif ((0 <= clay <= 8) and (25 <= silt <= 40) and (52 <= sand <= 75)):
        soil_class = "Su3"
    elif ((0 <= clay <= 8) and (40 <= silt <= 50) and (42 <= sand <= 60)):
        soil_class = "Su4"

    # --- Loam (L)
    elif ((8 <= clay <= 17) and (40 <= silt <= 50) and (33 <= sand < 52)):
        soil_class = "Slu"
    elif ((12 <= clay <= 17) and (10 <= silt <= 40) and (43 <= sand <= 78)):
        soil_class = "Sl4"
    elif ((17 <= clay <= 25) and (0 <= silt <= 15) and (60 <= sand <= 83)):
        soil_class = "St3"
    elif ((17 <= clay <= 25) and (40 <= silt <= 50) and (25 <= sand <= 43)):
        soil_class = "Ls2"
    elif ((17 <= clay <= 25) and (30 <= silt <= 40) and (35 <= sand <= 53)):
        soil_class = "Ls3"
    elif ((17 <= clay <= 25) and (15 <= silt <= 30) and (45 <= sand <= 68)):
        soil_class = "Ls4"
    elif ((25 <= clay <= 35) and (30 <= silt <= 50) and (15 <= sand <= 45)):
        soil_class = "Lt2"
    elif ((25 <= clay <= 45) and (15 <= silt <= 30) and (25 <= sand <= 60)):
        soil_class = "Lts"
    elif ((25 <= clay <= 35) and (0 <= silt <= 15) and (50 <= sand <= 75)):
        soil_class = "Ts4"
    elif ((35 <= clay <= 45) and (0 <= silt <= 15) and (40 <= sand <= 65)):
        soil_class = "Ts3"

    # --- Silt (U)
    elif ((0 <= clay <= 8) and (80 <= silt <= 100) and (0 <= sand <= 20)):
        soil_class = "Uu"
    elif ((0 <= clay <= 8) and (50 <= silt <= 80) and (12 <= sand <= 50)):
        soil_class = "Us"
    elif ((8 <= clay <= 12) and (65 <= silt <= 92) and (0 <= sand <= 27)):
        soil_class = "Ut2"
    elif ((12 <= clay <= 17) and (65 <= silt <= 88) and (0 <= sand <= 23)):
        soil_class = "Ut3"
    elif ((8 <= clay <= 17) and (50 <= silt <= 65) and (18 <= sand <= 42)):
        soil_class = "Uls"
    elif ((17 <= clay <= 25) and (65 <= silt <= 83) and (0 <= sand <= 18)):
        soil_class = "Ut4"
    elif ((17 <= clay <= 30) and (50 <= silt <= 65) and (5 <= sand <= 33)):
        soil_class = "Lu"

    # --- Clay (T)
    elif ((35 <= clay <= 45) and (30 <= silt <= 50) and (5 <= sand <= 35)):
        soil_class = "Lt3"
    elif ((30 <= clay <= 45) and (50 <= silt <= 65) and (0 <= sand <= 20)):
        soil_class = "Tu3"
    elif ((25 < clay <= 35) and (65 <= silt <= 75) and (0 <= sand <= 10)):
        soil_class = "Tu4"
    elif ((45 <= clay <= 65) and (0 <= silt <= 15) and (20 <= sand <= 55)):
        soil_class = "Ts2"
    elif ((45 <= clay <= 65) and (15 <= silt <= 30) and (5 <= sand <= 40)):
        soil_class = "Tl"
    elif ((45 <= clay <= 65) and (30 <= silt <= 55) and (0 <= sand <= 25)):
        soil_class = "Tu2"
    elif ((65 <= clay <= 100) and (0 <= silt <= 35) and (0 <= sand <= 35)):
        soil_class = "Tt"

    if soil_class == "":
        print(clay, silt, sand)

    return soil_class


def get_soil_texture_superclass(category_name: str):
    """Combine classes to superclasses.

    Parameters
    ----------
    category_name : str
        Name of the category

    """
    if category_name in ["Ss", "Su2", "Sl2", "Sl3", "St2", "Su3", "Su4"]:
        return "super_S"
    if category_name in ["Slu", "Sl4", "St3", "Ls2", "Ls3", "Ls4",
                         "Lt2", "Lts", "Ts4", "Ts3"]:
        return "super_L"
    if category_name in ["Uu", "Us", "Ut2", "Ut3", "Uls", "Ut4", "Lu"]:
        return "super_U"
    if category_name in ["Lt3", "Tu3", "Tu4", "Ts2", "Tl", "Tu2", "Tt"]:
        return "super_T"

    print("Warning: Category '{0}' was not included into superclasses."
          .format(category_name))
    return category_name


def get_soil_texture_superclass_id(superclass: str):
    """Get soil texture superclass ID.

    Parameters
    ----------
    superclass : str
        Superclass from {L, S, T, U}.

    Returns
    -------
    int
        ID of superclass

    """
    superclass_dict = {"L": 0, "S": 1, "T": 2, "U": 3}
    return superclass_dict[superclass[-1]]


def plot_confusion_matrix(cm,
                          classes: list,
                          ax,
                          normalize: bool = False,
                          title: str = 'Confusion matrix',
                          cmap=plt.cm.Blues,
                          fontsize: int = 10,
                          verbose=0,
                          show_xlabel: bool = True,
                          show_ylabel: bool = True):
    """Plot confusion matrix.

    Source: https://scikit-learn.org/0.18/auto_examples/
            model_selection/plot_confusion_matrix.html

    Parameters
    ----------
    cm : confusion matrix
        Matplotlib confusion matrix
    classes : list
        List of classes for the confusion matrix
    ax : axis
        Plot axis
    normalize : bool, optional (default=False)
        If true, the confusion matrix is normalized
    title : str, optional
        Title of the confusion matrix plot
    cmap : [type], optional
        Colormap for matplotlib
    fontsize : int, optional (default=10)
        Fontsize
    verbose : int, optional (default=0)
        Controls the verbosity.
    show_xlabel : bool, optional (default=True)
        If true, show label for x-axis
    show_ylabel : bool, optional (default=True)
        If true, show label for y-axis

    Returns
    -------
    img : plot
        Confusion matrix as plot

    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if verbose:
            print("Normalized confusion matrix")
    elif verbose:
        print('Confusion matrix, without normalization')

    if verbose:
        print(cm)

    img = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0., vmax=0.9)
    ax.set_title(title, fontsize=fontsize, weight="bold")
    # plt.colorbar(img, ax=ax)

    # set ticks
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    # set tick fontsizes
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    # set percentages
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=fontsize)

    # set axis labels
    if show_xlabel:
        ax.set_xlabel('Classified label', fontsize=fontsize)
    if show_ylabel:
        ax.set_ylabel('True label', fontsize=fontsize)

    return img
