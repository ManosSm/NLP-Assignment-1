import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import StrMethodFormatter

def make_data_frame(classifiers_data):
    """
    classifiers_data: a dict, where
        each key is an alias for a system
        which then maps to a list of runs
        
        each run is a dict with keys
        - encoder: this is the encoder used; there's one used for all folds;
        - models: the models trained; this is a list with one per training fold
        - predictions: the predictions of those models; this is a list with one array of predictions per classifier trained
        - accuracies: the validation accuracies of the trained classifiers; this is a list with one measurement per classifier trained
        
    """
    table = []
    for model_family, model_data in classifiers_data.items():
        for run, run_data in enumerate(model_data):
            for fold, acc in run_data['accuracies'].items():
                table.append([model_family, acc, run, fold])
    return pd.DataFrame(table, columns=['model', 'accuracy', 'run', 'fold'])


def plot_box_and_points(df, metric='accuracy', tickfmt="{x:.4f}"):
    """
    Plot the fold accuracies for various classifiers using one boxplot and a stripplot per classifier.

    df: a pandas DataFrame with two columns (model and metric)
    """     
    fig, ax = plt.subplots(figsize=(8, 4))
    order = df["model"].unique()
    
    sns.boxplot(
        data=df,
        x="model",
        y=metric,
        order=order,
        color="lightgray",
        showfliers=False,  # All measurements are shown by the strip plot
        ax=ax,
    )
    
    sns.stripplot(
        data=df,
        x="model",
        y=metric,
        hue='run',
        order=order,
        jitter=0.15,
        size=6,
        alpha=0.7,
        ax=ax,
    )
    
    ax.set(xlabel="model", ylabel=metric)
    ax.tick_params(axis="x", rotation=30)
    ax.yaxis.set_major_formatter(StrMethodFormatter(tickfmt))
    fig.tight_layout()
    plt.show()