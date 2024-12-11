import argparse
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix
import bradley_terry

######################################################################
# Reality check for Bradley-Terry implementation using synthetic data
######################################################################

# This function generates pairwise data by assigning a "true" win probability
# to each of num_items items as ground truth. Then for each pairwise "competition"
# between two items, it chooses each item's "score" from a Gaussian centered on
# that item's "true" win probability. The item in the pair with the higher score wins.
#
# For example, if item A has "true" win probability 0.4 and B has 0.7,
# the Gaussian sampling could lead to A's score being 0.45 and B's 0.65,
# in which case B wins this particular competition (consistent with its higher
# true win probability), or we could get A's score being 0.55 and B's 0.32,
# resulting in an "upset".
#
# The higher the variance, the more likely it will be that upsets will happen,
# i.e. the simulated data are noiser with respect to the "true" win probabilities.

def generate_pairwise_data(num_items, num_pairs, min_comparisons, variance, debug=False):
    
    items = [f"item_{i}" for i in range(num_items)]
    index2item = dict(enumerate(items))
    item2index = {v: k for k, v in index2item.items()}

    # "True" win probabilities
    # Rounding to 4 digits for easier readability when debugging
    win_probabilities = {item: round(random.uniform(0, 1), 4) for item in items}
    
    data = []
    win_count   = {item: 0 for item in items}
    total_count = {item: 0 for item in items}
    
    for _ in range(num_pairs):
        item1, item2 = random.sample(items, 2)
        score_item1  = np.random.normal(win_probabilities[item1], variance)
        score_item2  = np.random.normal(win_probabilities[item2], variance)
        
        if score_item1 > score_item2:
            data.append((item1, item2))  # item1 wins
            win_count[item1] += 1
        else:
            data.append((item2, item1))  # item2 wins
            win_count[item2] += 1
        
        total_count[item1] += 1
        total_count[item2] += 1

    excluded_items = [item for item in items if total_count[item] < min_comparisons]
    included_items = [item for item in items if total_count[item] >= min_comparisons]

    if debug:
        print(f"Items = {items}")
        print(f"'True' win_probabilities = {win_probabilities}")
        print(f"win_count   = {win_count}")
        print(f"total_count = {total_count}")
        print(f"excluded_items = {excluded_items}")
        print(f"included_items = {included_items}")
        print(f"original data ({len(data)} pairs) = {data}")

    return data, win_probabilities, win_count, total_count, excluded_items, included_items

def compute_manual_quantiles(values, quantiles):
    """Manually compute quantiles for a set of values."""
    sorted_indices = np.argsort(values)
    num_points = len(values)
    
    quantile_labels = np.zeros(num_points, dtype=int)
    points_per_quantile = num_points // quantiles

    for q in range(quantiles):
        start_idx = q * points_per_quantile
        if q == quantiles - 1:
            quantile_labels[sorted_indices[start_idx:]] = q
        else:
            quantile_labels[sorted_indices[start_idx:start_idx + points_per_quantile]] = q
    
    return quantile_labels

def plot_scatter(x_values, y_values, x_label, y_label, title, color, pdf, quantiles, add_jitter=False, included_items=[], highlight_outliers=False, regression=False):
    num_points = len(x_values)

    plt.figure()

    # Manually calculate the quantiles
    if quantiles:
        x_quantiles = compute_manual_quantiles(x_values, quantiles)
        y_quantiles = compute_manual_quantiles(y_values, quantiles)

    # Optionally add jitter to avoid point overlap
    if add_jitter:
        jitter_strength = 0.01 * (max(x_values) - min(x_values))
        x_values = x_values + np.random.normal(0, jitter_strength, len(x_values))
        y_values = y_values + np.random.normal(0, jitter_strength, len(y_values))

    # Prepare colors for points
    point_colors = [color] * num_points
    if quantiles and highlight_outliers:
            point_colors = ['red' if abs(x_q - y_q) > 1 else color for x_q, y_q in zip(x_quantiles, y_quantiles)]

    # Plot scatter
    sns.scatterplot(x=x_values, y=y_values, color=color, s=30)

    if not included_items:
        included_items = [f"({x:.2f},{y:.2f})" for x, y in zip(x_values, y_values)]

    # Annotate each point with its corresponding item (label)
    for i, (x, y) in zip(included_items, zip(x_values, y_values)):
        plt.text(x, y, i, fontsize=9, ha='right', va='bottom')

    # Optionally add regression line and confidence interval
    if regression:
        sns.regplot(x=x_values, y=y_values, color=color, ci=95, scatter=False)

    # Calculate Spearman correlation
    corr, p_value = spearmanr(x_values, y_values)
    
    # Add correlation info to the plot
    plt.text(0.05, 0.95, f"Spearman r = {corr:.4f}, p = {p_value:.4e}", 
             transform=plt.gca().transAxes, ha='left', va='top', fontsize=10, color='black')

    if quantiles:

        # Shade the diagonal quantile regions with light blue and add dotted borders
        x_boundaries = np.percentile(x_values, np.linspace(0, 100, quantiles + 1))
        y_boundaries = np.percentile(y_values, np.linspace(0, 100, quantiles + 1))


        # Iterate over quantiles
        for i in range(quantiles):
            for j in range(quantiles):
                # Only shade when x and y are in the same quantile
                if i == j:
                    plt.fill_betweenx([y_boundaries[i], y_boundaries[i + 1]],  # y-range for the quantile
                                      x_boundaries[j], x_boundaries[j + 1],    # x-range for the quantile
                                      color='lightblue', alpha=0.3)            # Shade in light blue

                # Add dotted lines for each boundary
                plt.plot([x_boundaries[j], x_boundaries[j]], 
                         [y_boundaries[0], y_boundaries[-1]], 'b:', linewidth=1)  # Vertical dotted line
                plt.plot([x_boundaries[j + 1], x_boundaries[j + 1]], 
                         [y_boundaries[0], y_boundaries[-1]], 'b:', linewidth=1)  # Vertical dotted line
                plt.plot([x_boundaries[0], x_boundaries[-1]], 
                         [y_boundaries[i], y_boundaries[i]], 'b:', linewidth=1)    # Horizontal dotted line
                plt.plot([x_boundaries[0], x_boundaries[-1]], 
                         [y_boundaries[i + 1], y_boundaries[i + 1]], 'b:', linewidth=1)  # Horizontal dotted line

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    pdf.savefig()
    plt.close()

    if quantiles:
        create_confusion_matrix(x_quantiles, y_quantiles, x_label, y_label, title, pdf)
        

def create_confusion_matrix(x_quantiles, y_quantiles, x_label, y_label, title, pdf):
    num_points = len(x_quantiles)
    print(f"Creating confusion matrix for {num_points} points.")

    cm = confusion_matrix(x_quantiles, y_quantiles)
    cm_percentage = cm / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=True, square=True,
                xticklabels=[f"Q{i+1}" for i in range(len(cm))],
                yticklabels=[f"Q{i+1}" for i in range(len(cm))])
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{title} - Quantile Confusion Matrix", fontsize=10)
    pdf.savefig()
    plt.close()


def test_bradley_terry(num_items, num_pairs, min_comparisons, variance, output_pdf, quantiles, regression, verbose=False, debug=False):
    
    with PdfPages(output_pdf) as pdf:

        # Generating a simulated dataset for testing
        print("Generating pairwise data...")
        pairwise_data_result = generate_pairwise_data(num_items, num_pairs, min_comparisons, variance, debug=debug)
        data, win_probabilities, win_count, total_count, excluded_items, included_items = pairwise_data_result 
        
        # Remove data pairs for items that have been excluded because they occur in fewer than min_comparisons comparisons
        if excluded_items:
            print(f"Excluded {len(excluded_items)} items due to too-few comparisons (less than {min_comparisons})")
            
        filtered_data             = [(item1, item2) for item1, item2 in data if item1 in included_items and item2 in included_items]
        simulated_win_proportions = {item: win_count[item] / total_count[item] for item in included_items}


        if debug:
            print(f"Simulated win proportions = {simulated_win_proportions}",flush=True)

        print("Running Bradley-Terry...")
        params = bradley_terry.run_bradley_terry(filtered_data, verbose=verbose)
        
        if verbose:
            print(f"Estimated strengths: {params}")

        print("Creating plots...")

        plot_scatter(
            [win_probabilities[i]         for i in included_items],
            [simulated_win_proportions[i] for i in included_items],
            'True Win Probability',
            f'Simulated Win Proportion (variance={variance})',
            f"Simulated Win Proportion vs 'True' Win Probability",
            'blue',
            pdf,
            quantiles,
            add_jitter=False,
            highlight_outliers=True,
            regression=regression,
            included_items = included_items
        )

        plot_scatter(
            [simulated_win_proportions[i] for i in included_items],
            [params[i]                    for i in included_items],
            f'Simulated Win Proportion (variance={variance})',
            'Bradley-Terry Parameter',
            "Bradley-Terry Parameters vs Simulated Win Proportions",
            'purple',
            pdf,
            quantiles,
            add_jitter=False,
            highlight_outliers=True,
            regression=regression,
            included_items = included_items
        )
        
        plot_scatter(
            [win_probabilities[i] for i in included_items],
            [params[i]            for i in included_items],
            'True Win Probability',
            'Bradley-Terry Parameter',
            "Bradley-Terry Parameters vs 'True' Win Probabilities",
            'green',
            pdf,
            quantiles,
            add_jitter=False,
            highlight_outliers=True,
            regression=regression,
            included_items = included_items
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bradley-Terry model test and save output to a PDF.")
    parser.add_argument('--num_items', type=int, default=20, help='Number of items')
    parser.add_argument('--num_pairs', type=int, default=50, help='Number of pairwise comparisons')
    parser.add_argument('--min_comparisons', type=int, default=3, help='Minimum pairwise comparisons required to include an item')
    parser.add_argument('--variance', type=float, default=0.1, help='Variance for Gaussian distribution')
    parser.add_argument('--output', type=str, default="out.pdf", help='Output PDF filename')
    parser.add_argument('--regression', action='store_false', help='Include regression line and confidence interval (defaults to True)')
    parser.add_argument('--verbose', action='store_true', help='Include verbose output (defaults to False)')
    parser.add_argument('--debug', action='store_true', help='Include debug output (defaults to False)')

    args = parser.parse_args()

    # Debugging will be verbose by definition
    if args.debug:
        args.verbose = True
        
    test_bradley_terry(args.num_items, args.num_pairs, args.min_comparisons, args.variance, args.output, False, args.regression,
                           verbose=args.verbose, debug=args.debug)
    
    print(f"\nOutput is in file {args.output}")
    print( f"\nThe first plot, Simulated Win Proportion vs 'True' Win Probability\n",
              f"shows how much noise has been added in the simulated competitions by using variance = {args.variance}")
    print( f"\nThe second plot, Bradley-Terry Parameters vs Simulated Win Proportions\n",
              f"shows how well Bradley-Terry has recovered the observed (with noise) item-level proportions of wins.")
    print( f"\nThe third plot of Bradley-Terry Parameters vs Simulated Win Proportions\n",
              f"shows how well Bradley-Terry has recovered the underlying ground truth for items' win probabilities, even with noise added.\n")
