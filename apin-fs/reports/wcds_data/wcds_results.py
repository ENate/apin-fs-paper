import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# colors
colors_accuracies = [
    "purple",
    "green",
    "orange",
    "magenta",
    "#CFC60E",
    "#0FBBAE",
    "blue",
]

colors_inputs = ["purple", "green", "orange", "magenta", "#CFC60E", "#0FBBAE", "blue"]
# Testing set


def plot_wcds(kwarg_dicts, tp_fp, tn_fn):
    # plotting parameters
    selected_inputs = kwarg_dicts.get("optimal_inputs")
    accuracies = kwarg_dicts.get("training_accuracy")
    accuracies_testing = kwarg_dicts.get("testing_accuracy")
    error_dict = kwarg_dicts.get("errors")
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    #
    # Selected Inputs
    plt.subplot(2, 2, 1)  # 1 row, 2 cols, subplot 2
    plt.ylabel("Selected Inputs")
    sns.barplot(
        x=list(selected_inputs.keys()),
        y=list(selected_inputs.values()),
        palette=colors_inputs,
    )

    # Error plots
    # plt.subplot(2, 3, 2)  # 1 row, 2 cols, subplot 2
    # plt.ylabel("Error")
    # sns.barplot(x=list(error_dict.keys()), y=list(
    #    error_dict.values()), palette=colors_inputs)

    # Training accuracy
    # plt.subplot(2, 3, 3)  # 1 row, 2 cols, subplot 2
    # plt.ylabel("Training Accuracy")
    # plt.xlabel("Algorithms")
    # sns.barplot(x=list(accuracies.keys()), y=list(
    #    accuracies.values()), palette=colors_inputs)

    # Testing accuracy
    plt.subplot(2, 2, 2)  # 1 row, 2 cols, subplot 2
    plt.ylabel("Testing Accuracy")
    plt.xlabel("Algorithms")
    sns.barplot(
        x=list(accuracies_testing.keys()),
        y=list(accuracies.values()),
        palette=colors_inputs,
    )
    plt.yticks(np.arange(0, 100, 10))

    # False positives
    # Testing accuracy
    plt.subplot(2, 2, 3)  # 1 row, 2 cols, subplot 2
    plt.ylabel("TP - TN")

    plt.xlabel("Algorithms")
    # sns.barplot(x=list(accuracies_testing.keys()), y=list(
    #    accuracies.values()), palette=colors_inputs)
    plt.yticks(np.arange(0, 100, 10))
    sns.barplot(x="group", y="y_quantity", hue="x_quantity", data=tp_fp)

    # True Positive & True Negative
    # Testing accuracy
    plt.subplot(2, 2, 4)  # 1 row, 2 cols, subplot 2
    plt.ylabel("FP - FN")
    plt.xlabel("Algorithms")
    # sns.barplot(x=list(accuracies_testing.keys()), y=list(
    #    accuracies.values()), palette=colors_inputs)
    sns.barplot(x="group", y="y_qty", hue="x_qty", data=tn_fn)
    # save a pdf file
    fig.savefig("~compare_wcds.pdf")
    # save an eps file
    fig.savefig(
        "compare_wcds.pdf"
    )  # eps', format='eps', dpi=800)
    plt.show()


def plot_all_data():
    pass


if __name__ == "__main__":
    accuracies = {
        "RF": 92.98245614035088,
        "OA": 95.32163742690058,
        "L1": 95.32163742690058,
        "EN": 95.90643274853801,
        "DP": 94.15204678362574,
        "XbC": 88.90643274853801,
    }

    accuracies_testing = {
        "RF": 92.98245614035088,
        "DP": 95.32163742690058,
        "EN": 95.90643274853801,
        "L1": 94.15204678362574,
        "XbC": 88.90643274853801,
        "OA": 95.32163742690058,
    }
    selected_inputs = {"RF": 8, "L1": 32, "XbC": 8, "DP": 32, "EN": 32, "OA": 6}
    error_dict = {
        "RF": 0.08,
        "LGM": 0.07,
        "XbC": 0.08,
        "CaB": 0.09,
        "EN": 0.05,
        "OA": 0.06,
    }
    plotting_args = {
        "training_accuracy": accuracies,
        "testing_accuracy": accuracies_testing,
        "optimal_inputs": selected_inputs,
        "errors": error_dict,
    }
    # True Positives and False Negatives
    df_TP_FP = {
        "x_quantity": [
            "TP",
            "FP",
            "TP",
            "FP",
            "TP",
            "FP",
            "TP",
            "FP",
            "TP",
            "FP",
            "TP",
            "FP",
        ],
        "y_quantity": [61, 4, 62, 3, 60, 5, 63, 2, 63, 2, 63, 2],
        "group": [
            "RF",
            "RF",
            "L1",
            "L1",
            "XbC",
            "XbC",
            "EN",
            "EN",
            "DP",
            "DP",
            "OA",
            "OA",
        ],
    }
    df_TN_FN = {
        "x_qty": [
            "TN",
            "FN",
            "TN",
            "FN",
            "TN",
            "FN",
            "TN",
            "FN",
            "TN",
            "FN",
            "TN",
            "FN",
        ],
        "y_qty": [22, 4, 21, 5, 23, 3, 21, 5, 21, 5, 24, 2],
        "group": [
            "RF",
            "RF",
            "L1",
            "L1",
            "XbC",
            "XbC",
            "EN",
            "EN",
            "DP",
            "DP",
            "OA",
            "OA",
        ],
    }
    plot_wcds(plotting_args, df_TP_FP, df_TN_FN)
