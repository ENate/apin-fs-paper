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
    """Plotting comparisons between different methods
    Args:
        kwarg_dicts ([type]): [array of testing and training accuracy]
        tp_fp ([type]): [array of true and false positive values]
        tn_fn ([type]): [array of true and false positive values]
    """
    # plotting parameters
    selected_inputs = kwarg_dicts.get("optimal_inputs")
    # Maybe already plotted
    # accuracies = kwarg_dicts.get('training_accuracy')
    accuracies_testing = kwarg_dicts.get("testing_accuracy")
    # Check if plotted already
    # error_dict = kwarg_dicts.get('errors')
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
    # plt.subplot(2, 2, 3)  # 1 row, 2 cols, subplot 2
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
        y=list(accuracies_testing.values()),
        palette=colors_inputs,
    )

    # False positives
    # Testing accuracy
    plt.subplot(2, 2, 3)  # 1 row, 2 cols, subplot 2
    plt.ylabel("TP - TN")

    plt.xlabel("Algorithms")
    # sns.barplot(x=list(accuracies_testing.keys()), y=list(
    #    accuracies.values()), palette=colors_inputs)
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
    fig.savefig(
        "compare_hdata.pdf"
    )
    # save an eps file
    fig.savefig(
        "compare_hdata.pdf"
    )
    # eps', format='eps', dpi=800)
    plt.show()


def plot_all_data():
    """oops"""
    pass


if __name__ == "__main__":
    accuracies = {
        "RF": 92.088,
        "LG": 95.901,
        "DP": 95.358,
        "NB": 95.358,
        "EN": 95.901,
        "L1": 94.574,
        "XbC": 95.801,
    }

    accuracies_testing = {
        "RF": 65.98,
        # 'LG': 53.90,
        "DP": 55.32,
        "NB": 50.32,
        "EN": 61.90,
        "L1": 60.14,
        "XbC": 69.91,
        "OA": 57.90,
    }
    selected_inputs = {"RF": 30, "L1": 35, "XbC": 32, "DP": 35, "EN": 35, "OA": 25}
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
        "y_quantity": [89, 29, 85, 23, 88, 20, 90, 18, 94, 14, 98, 10],
        "group": [
            "RF",
            "RF",
            "DP",
            "DP",
            "XbC",
            "XbC",
            "L1",
            "L1",
            "EN",
            "EN",
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
        "y_qty": [41, 22, 43, 20, 45, 18, 47, 16, 48, 15, 50, 13],
        "group": [
            "RF",
            "RF",
            "XbC",
            "XbC",
            "DP",
            "DP",
            "L1",
            "L1",
            "EN",
            "EN",
            "OA",
            "OA",
        ],
    }
    plot_wcds(plotting_args, df_TP_FP, df_TN_FN)
