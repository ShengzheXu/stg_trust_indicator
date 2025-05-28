import matplotlib.pyplot as plt

def plot_inter_cluster_cosine_by_layer(dfs, legend_out=False, is_with_cluster=True, save_path=None):
    """
    Plots Inter-Cluster Cosine by Layer ID for each DataFrame in the list.
    Styled to match the provided plot formatting.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for df in dfs:
        x = df.index
        if is_with_cluster:
            y = df["Inter-Cluster Cosine"]
        else:
            y = df["Inter-Cluster Cosine (No Cluster)"]
        label = f"NMSE: {df.iloc[0]['NMSE']:.4f}"
        ax.plot(x, y, linewidth=3, marker='o', label=label)

    if is_with_cluster:
        ax.set_title('Inter-Cluster Cosine (w/ Cluster) by Layer', fontsize=24)
    else:
        ax.set_title('Inter-Cluster Cosine (No Cluster) by Layer', fontsize=24)
    ax.set_xlabel('Layer ID', fontsize=24)
    ax.set_ylabel('Inter-Cluster Cosine', fontsize=24)
    ax.legend(fontsize=16, loc='upper left')
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xticks(range(0, len(x), 2))
    
    if legend_out:
        ax.legend(fontsize=16, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(right=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    # close the figure to free up memory
    plt.close(fig)


def plot_kmeans_by_layer(dfs, legend_out=False, is_with_cluster=True, save_path=None):
    """
    Plots Inter-Cluster Cosine by Layer ID for each DataFrame in the list.
    Styled to match the provided plot formatting.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for df in dfs:
        x = df.index
        if is_with_cluster:
            y = df["Best K"]
        else:
            y = df["Best K (No Cluster)"]
        label = f"NMSE: {df.iloc[0]['NMSE']:.4f}"
        ax.plot(x, y, linewidth=3, marker='o', label=label)

    if is_with_cluster:
        ax.set_title('K-Cluster (w/ Cluster) by Layer', fontsize=24)
    else:
        ax.set_title('Inter-Cluster Cosine (No Cluster) by Layer', fontsize=24)
    ax.set_xlabel('Layer ID', fontsize=24)
    ax.set_ylabel('Best K', fontsize=24)
    ax.legend(fontsize=16, loc='upper left')
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=18)
    # x axis ticks only integers
    ax.set_xticks(range(0, len(x), 2))

    if legend_out:
        ax.legend(fontsize=16, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(right=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    # close the figure to free up memory
    plt.close(fig)

import os
import pandas as pd

def after_check(folder):
    exp_name = folder.split("/")[-1]
    print(f"After check for {folder}, exp_name: {exp_name}")
    # file_path = folder + sub_folders + "/output_isotropy_metric/" + sub_folders + "_exp01.csv"

    file_path = os.path.join(folder, "output_isotropy_metric", f"{exp_name}_exp01.csv")
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return 
    else:
        print(f"File {file_path} exists.")
        df = pd.read_csv(file_path)

        # save to folder + sub_folders + "/output_plots/"
        output_folder = os.path.join(folder, "output_plots")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plot_inter_cluster_cosine_by_layer([df], is_with_cluster=True, save_path=os.path.join(output_folder, f"{exp_name}_inter_cluster_cosine.png"))
        plot_inter_cluster_cosine_by_layer([df], is_with_cluster=False, save_path=os.path.join(output_folder, f"{exp_name}_inter_cluster_cosine_no_cluster.png"))
        plot_kmeans_by_layer([df], is_with_cluster=True, save_path=os.path.join(output_folder, f"{exp_name}_kmeans.png"))


if __name__ == "__main__":
    exp_folder = f"results/single_exp_{50}in_{200}out_100rows_{'chronos-t5-small'}"
    after_check(exp_folder)