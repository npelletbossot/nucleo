"""
nucleo.merging_function
------------------------
Merging parquet files in order to calculate heatmaps.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import os
import pickle
import collections
import json

import numpy as np
import polars as pl
from pathlib import Path
from tqdm import tqdm
from dataclasses import fields


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


# 2.1 : Dialog box 


def ask_confirmation_input():
    response = input("Are you sure you want to merge all the parquet files? (Yes/No): ")
    if response != "Yes":
        raise RuntimeError("Stopped by the user.")


# 2.2 : Merging into one file


def merging_parquet_files(root_directory, nt, tmax, dt, alphao, alphaf, beta, Lmin, Lmax, origin, bps, SimulationData):
    """
    Merges all the .parquet files from our simulations while filtering only columns of types str, i64, and f64.

    Args:
        root_directory (str): Root directory containing subfolders.
        L_max (int): Filter parameter for subfolders.
        n_t (int): Filter parameter for subfolders.

    Returns:
        None
    """

    os.chdir(root_directory)  # Navigate to root directory to avoid unwanted paths
    print("Launched at:", os.getcwd())

    # Main result storage
    dataframes = []

    # List of subfolders to check in the root folder
    subdirs = [os.path.join(root_directory, d) for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]

    # Main loop
    for subdir in tqdm(subdirs, desc="Loading subfiles", unit=" subfiles"):
        for dirpath, dirnames, filenames in os.walk(subdir):
            for dirname in dirnames:
                folder_name = dirname
                # Verification
                if (f"nt_{nt}_" in folder_name and 
                    f"tmax_{tmax}" in folder_name and 
                    f"dt_{dt}" in folder_name and 
                    f"alphao_{alphao}" in folder_name and 
                    f"alphaf_{alphaf}" in folder_name and 
                    f"beta_{beta}" in folder_name and 
                    f"Lmin_{Lmin}" in folder_name and 
                    f"Lmax_{Lmax}" in folder_name and 
                    f"origin_{origin}" in folder_name and 
                    f"bps_{bps}" in folder_name 
                    ):
                    # Path of the corresponding subfolder
                    subdir_path = os.path.join(dirpath, folder_name)

                    for filename in os.listdir(subdir_path):
                        if filename.endswith('.parquet') and 'data' in filename:
                            pq_path = os.path.join(subdir_path, filename)

                            try:
                                # Read the parquet file
                                df_parquet = pl.read_parquet(pq_path)
                                df_columns = df_parquet.columns
                                class_columns = set(f.name for f in fields(SimulationData))

                                if collections.Counter(df_columns) == collections.Counter(class_columns):

                                    # Filter only columns of type str, i64, or f64
                                    filtered_df = df_parquet.select([
                                        col for col, dtype in zip(df_parquet.columns, df_parquet.dtypes)
                                        if dtype in [pl.Utf8, pl.Int64, pl.Float64]
                                    ])
                                    
                                    dataframes.append(filtered_df)

                            except Exception as e:
                                print(f"Error loading Parquet file: {e}")
    
    # Concatenate and write the final result
    if dataframes:
        merged_df = pl.concat(dataframes)
        last_address = f"{root_directory}/ncl_output.parquet"
        merged_df.write_parquet(last_address)
        print(f"All files merged and saved to {last_address}.")
    else:
        print("No Parquet files found that match the criteria.")

    return merged_df


# 2.3 : Getting and ordering configurations


def getting_and_ordering_configurations(data_frame, scenario_path = Path.home() / "Documents" / "PhD" / "Workspace" / "nucleo" / "outputs" / "2025-01-01_PSMN"): 
    """
    We're extracting the different configurations of modeling and ordering them for a proper representation

    Args:
        data_frame (df): filtered data frame

    Returns:
        sorted_combinations_configs: the configurations
    """

    df = data_frame                             # More convinient
    filtered_combinations = df.filter(
        ~(
            ((pl.col("landscape") == 'periodic') | (pl.col("landscape") == 'homogeneous')) &
            (pl.col("bpmin") == 5)
        )
    )

    # Getting the unique combinations of 's', 'l', 'bpmin' and 'landscape'
    unique_combinations = filtered_combinations.select(['s', 'l', 'bpmin', 'landscape']).unique()

    # Ordering it by landscape in priority
    alpha_order = pl.when(pl.col("landscape") == 'homogeneous').then(1)\
                    .when(pl.col("landscape") == 'periodic').then(2)\
                    .when(pl.col("landscape") == 'random').then(3)\
                    .otherwise(4)
    unique_combinations = unique_combinations.with_columns(
        alpha_order.alias("alpha_order")
    )

    # Ordering by 'alpha_order', then by 'bpmin', and finally by 'l' with l=10 prioritazed
    sorted_combinations = unique_combinations.sort(by=['alpha_order', 'bpmin', 'l'])

    # Suppressing the temporary column 'alpha_order'
    sorted_combinations = sorted_combinations.drop('alpha_order')

    # Convertiing it into a list of dict
    sorted_combinations_configs = sorted_combinations.rows()
    sorted_combinations_configs = [
        {"s": row[0], "l": row[1], "bpmin": row[2], "landscape": row[3]} 
        for row in sorted_combinations_configs
    ]

    with open("configs.json", "w") as f:
        json.dump(sorted_combinations_configs,f)

    # Print
    for config in sorted_combinations_configs:
        print(config)

    with open(f"{scenario_path}/scenarios.json", "w") as f:
        json.dump(sorted_combinations_configs,f)

    return sorted_combinations_configs


# 2.4 : From main file to heatmaps


def compute_heatmap_data(df: pl.DataFrame, config_list: list, speed_cols: list, root: str) -> dict:
    """
    Computes and stores heatmap data for each configuration and speed column.
    Linear speeds are getting normalized by the theoretical values of constant_mean scenario.

    Args:
        df (pl.DataFrame): Data containing multiple speed configurations.
        config_list (list): List of configuration dictionaries.
        speed_cols (list): List of speed columns to process.

    Returns:
        dict: Nested dictionary {config_idx -> {speed_col -> heatmap_data, config_metadata}}.
    """

    all_data_raw = {}
    all_data_norm_mu = {}
    all_data_norm_th = {}

    for idx, config in tqdm(enumerate(config_list), total=len(config_list), desc="Computing heatmaps"):

        df_filtered = df.filter(
            (pl.col('s') == config['s']) &
            (pl.col('l') == config['l']) &
            (pl.col('bpmin') == config['bpmin']) &
            (pl.col('landscape') == config['landscape'])
        )

        if df_filtered.is_empty():
            print(f"No data for config {config}")
            continue

        mu_values = df_filtered['mu'].unique().sort()
        theta_values = df_filtered['theta'].unique().sort()

        # Store configuration metadata (init both dicts)
        meta = {
            "mu_values": mu_values,
            "theta_values": theta_values,
            "config": config
        }

        all_data_raw[idx] = dict(meta)
        all_data_norm_mu[idx] = dict(meta)
        all_data_norm_th[idx] = dict(meta)

        # Loop on data
        for speed_col in speed_cols:
            heatmap_raw = []
            heatmap_norm_mu = []
            heatmap_norm_th = []

            for theta in tqdm(theta_values, desc=f"Processing theta for config {idx}", leave=False):
                data_raw = []
                data_norm_mu = []
                data_norm_th = []

                for mu in mu_values:
                    values = df_filtered.filter((pl.col('mu') == mu) & (pl.col('theta') == theta))[speed_col].mean()
                    norm_mu = mu
                    norm_th = ((df_filtered['alphao'][0] * df_filtered['s'][0] +
                                df_filtered['alphaf'][0] * df_filtered['l'][0]) /
                                (df_filtered['l'][0] + df_filtered['s'][0])) * mu

                    if speed_col in {'v_mean', 'vi_med', 'vi_mp', 'vf'}:
                        data_raw.append(values if values is not None else 0)
                        data_norm_mu.append(values / norm_mu if values is not None else 0)
                        data_norm_th.append(values / norm_th if values is not None else 0)
                       
                    elif speed_col in {'Cf', 'wf'}:
                        data_raw.append(values if values is not None else 0)
                        data_norm_mu.append(values if values is not None else 0)
                        data_norm_th.append(values if values is not None else 0)

                heatmap_raw.append(data_raw)
                heatmap_norm_mu.append(data_norm_mu)
                heatmap_norm_th.append(data_norm_th)

            all_data_raw[idx][speed_col] = np.array(heatmap_raw)
            all_data_norm_mu[idx][speed_col] = np.array(heatmap_norm_mu)
            all_data_norm_th[idx][speed_col] = np.array(heatmap_norm_th)

    # Saving datas
    path_raw  = os.path.join(root, "hmp_nucleo_raw.pkl")
    path_norm_mu = os.path.join(root, "hmp_nucleo_nmu.pkl")
    path_norm_th = os.path.join(root, "hmp_nucleo_nth.pkl")

    with open(path_raw, "wb") as f:
        pickle.dump(all_data_raw, f)
    with open(path_norm_mu, "wb") as f:
        pickle.dump(all_data_norm_mu, f)
    with open(path_norm_th, "wb") as f:
        pickle.dump(all_data_norm_th, f)

    print("Files written at :", path_raw, "-", path_norm_mu, "-", path_norm_th)
    return None


# ─────────────────────────────────────────────
# 3 : Call
# ─────────────────────────────────────────────


# 3.0 : Root
root = Path.home() / "Documents" / "PhD" / "Workspace" / "nucleo" / "outputs" / "2025-01-01_PSMN"
root_parquet = root / "ncl_output.parquet"


# 3.1 : Asking confirmation
ask_confirmation_input()


# 3.2 : Merging
merged_df = merging_parquet_files(root_directory=root, nt=10000, tmax=100, dt=1, alphao=0, alphaf=1, beta=0, Lmin=0, Lmax=50000, origin=10000, bps=1)
merged_df = pl.read_parquet(root_parquet)
print(merged_df.head(5))


# 3.3 : Configurations
print('\nConfigurations :')
sorted_combinations_configs = getting_and_ordering_configurations(merged_df, root)


# 3.4 : Comuting heatmaps
speed_columns = ['v_mean', 'vi_med', 'vi_mp', 'vf', 'Cf', 'wf']
compute_heatmap_data(merged_df, sorted_combinations_configs, speed_columns, root)
print("Heatmaps computed")