import matplotlib
matplotlib.use("Agg")
import numpy as np
from run.utils.triviaqa_evaluation import exact_match_score, f1_score, metric_max_over_ground_truths, normalize_answer
import matplotlib.pyplot as plt
import torch

def get_no_heads_layers(entropy_vector):
    if type(entropy_vector) == np.ndarray:
        pass
    elif type(entropy_vector) == torch.Tensor:
        entropy_vector = entropy_vector.cpu().numpy()

    if entropy_vector.shape[0] == 4096:
        return 64, 64
    elif entropy_vector.shape[0] == 2560:
        return 40, 64
    elif entropy_vector.shape[0] == 256:
        return 16, 16
    elif entropy_vector.shape[0] == 1152:
        return 36, 32
    elif entropy_vector.shape[0] == 448:
        return 28, 16
    elif entropy_vector.shape[0] == 1024:
        return 32, 32
    elif entropy_vector.shape[0] == 672:
        return 28, 24
    elif entropy_vector.shape[0] == 768:
        return 48, 16 
    else:
        print("Not the right shape, entropy_vector.shape: ", entropy_vector.shape)
        print("Skipping heatmap plot")
        raise ValueError("Not the right shape")
    
def plot_entropy_heatmap(entropy_vector, fname, title, center_zero=False, vlimits=None):
    n_layers, n_heads = get_no_heads_layers(entropy_vector)
    entropy_matrix = entropy_vector.reshape(n_layers, n_heads)

    plt.figure(figsize=(12, 8))
    if center_zero:
        if vlimits is not None:
            vmin = vlimits[0]
            vmax = vlimits[1]
        else:
            vmax = np.abs(entropy_matrix).max()
            vmin = -vmax
    else:
        vmin = None
        vmax = None

    plt.imshow(
        entropy_matrix,
        aspect="auto",
        cmap="bwr", #if center_zero else "viridis",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(
        label="Entropy"
    )
    plt.title(title)
    plt.xlabel(f"Head (0-{n_heads-1})")
    plt.ylabel(f"Layer (0-{n_layers-1})")
    plt.xticks(np.arange(0, n_heads, 8))
    plt.yticks(np.arange(0, n_layers, 5))
    plt.tight_layout()
    plt.savefig(fname)
    print(f"Saved heatmap to {fname}")
    plt.close()



def make_histograms(
    data_1,
    data_2,
    *,
    bins=50,
    name1="true",
    name2="false",
    x_label="entropy",
    fname="entropy_histograms.png",
    colors=("tab:blue", "tab:orange"),
):
    plt.figure(figsize=(8, 6))

    plt.hist(
        data_1,
        bins=bins,
        alpha=0.5,
        label=f"{name1} ({len(data_1)})",
        color=colors[0],
        edgecolor="k",
        density=True,  # Normalize to area = 1
    )
    plt.hist(
        data_2,
        bins=bins,
        alpha=0.5,
        label=f"{name2} ({len(data_2)})",
        color=colors[1],
        edgecolor="k",
        density=True,  # Normalize to area = 1
    )

    plt.xlabel(x_label)
    plt.ylabel("density")  # Not "frequency" since we're normalizing
    plt.title(f"{x_label} distribution: {name1} vs {name2}")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{fname}.png", dpi=300)
    plt.close()
    print(f"Saved → {fname}.png")


def make_line(
    mean_t, mean_f, fname="features_distribution.png", name1="true", name2="false"
):
    x = np.arange(len(mean_t))  # Assume 1 value per head

    plt.figure(figsize=(10, 6))
    plt.plot(x, mean_t, label=f"{name1} mean", marker="o", linestyle="-")
    plt.plot(x, mean_f, label=f"{name2} mean", marker="x", linestyle="--")

    plt.xlabel("Head Index")
    plt.ylabel("Mean Value")
    plt.title("Feature Mean per Head")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved → {fname}")


def make_scatter_with_lines(
    mean_t,
    mean_f,
    fname="features_distribution.png",
    name1="true",
    name2="false",
    metric="Mean",
):
    x = np.arange(len(mean_t))  # Head indices

    plt.figure(figsize=(10, 6))

    # Plot scatter points
    plt.scatter(x, mean_t, label=f"{name1} mean", marker="o", color="blue")
    plt.scatter(x, mean_f, label=f"{name2} mean", marker="x", color="red")

    # Draw a line between the two means for each head
    for i in x:
        plt.plot(
            [i, i], [mean_t[i], mean_f[i]], color="gray", linestyle="--", alpha=0.6
        )

    plt.xlabel("Head Index")
    plt.ylabel(f"{metric} Value")
    plt.title(f"Feature {metric} per Head")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved → {fname}")

def make_mean_variance_scatter(
    mean_t,
    var_t,
    mean_f,
    var_f,
    fname="mean_variance_scatter.png",
    name1="true",
    name2="false",
):
    plt.figure(figsize=(10, 6))
    
    # Plot group 1 (true)
    plt.scatter(mean_t, var_t, label=f"{name1}", marker="o", color="blue", alpha=0.7)

    # Plot group 2 (false)
    plt.scatter(mean_f, var_f, label=f"{name2}", marker="x", color="red", alpha=0.7)

    plt.xlabel("Mean Shapley Value")
    plt.ylabel("Variance of Shapley Value")
    plt.title("Mean vs Variance per Head")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved → {fname}")

def get_slopes(data):
    num_examples, num_features = data.shape
    x = np.arange(num_features).reshape(-1, 1)  # Feature indices
    slopes = []

    for i in range(num_examples):
        y = data[i]
        model = LinearRegression().fit(x, y)
        slopes.append(model.coef_[0])

    return np.array(slopes)


def plot_slopes(data_t, data_f, fname="entropy_slope_comparison.png"):
    slopes_t = get_slopes(data_t)
    slopes_f = get_slopes(data_f)

    plt.figure(figsize=(8, 6))
    plt.hist(
        slopes_t,
        bins=50,
        alpha=0.5,
        label=f"true ({len(slopes_t)})",
        color="tab:blue",
        density=True,
    )
    plt.hist(
        slopes_f,
        bins=50,
        alpha=0.5,
        label=f"false ({len(slopes_f)})",
        color="tab:orange",
        density=True,
    )

    plt.xlabel("Slope of entropy trend across feature index")
    plt.ylabel("Density")
    plt.title("Entropy slope: true vs false")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved → {fname}")

def make_top_low_histograms(X_test_raw, y_test, idxs_high=None, idxs_low=None, out_path="."):
    if not idxs_high:
        idxs_high = [200, 222, 248, 281, 249]
    if not idxs_low:
        idxs_low = [396, 388, 71, 353, 354]

    for i, head in enumerate(list(idxs_high)):
        column = X_test_raw[:,head]
        data_t = column[y_test == True]
        data_f = column[y_test == False]
        make_histograms(
            data_t, data_f, fname=f"{out_path}/feature_top_{i}_{head}", x_label="entropy",
        )         
    for i, head in enumerate(list(idxs_low)):
        column = X_test_raw[:,head]
        data_t = column[y_test == True]
        data_f = column[y_test == False]
        make_histograms(
            data_t, data_f, fname=f"{out_path}/feature_low_{i}_{head}", x_label="entropy",
        )