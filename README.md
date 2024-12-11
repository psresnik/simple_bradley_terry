
# Simple Bradley-Terry

A simple wrapper designed to make it very easy to use the [Bradley-Terry (1952)](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) model to transform a set of paired comparisons to a set of scores ordering the items being compared.

### How to run

```
import bradley_terry

scores = run_bradley_terry(pairs, alpha=0.1, verbose=False)
```

The `pairs` argument is a list of pairs where the first item in the pair is the "winner" for that comparison. For example:

```
pairs = [('item_1', 'item_0'), ('item_3', 'item_1'),  ...etc]
```

This implementation uses the [choix](https://github.com/lucasmaystre/choix) package. See the [choix discussion of pairwise comparisons for details](https://github.com/lucasmaystre/choix/blob/master/notebooks/intro-pairwise.ipynb), including the use of alpha for regularization. From the [choix documentation](https://web.archive.org/web/20240414140923/https://choix.lum.li/_/downloads/en/latest/pdf/): 

> Inference functions in choix provide a generic regularization argument: alpha. When 𝛼 = 0, regularization is turned off; setting 𝛼 > 0 turns it on. In practice, if , if regularization is needed, we recommend starting with small values (e.g., 10^{−4}) and increasing the value if necessary. ... In the special case of pairwise-comparison data, this can be loosely understood as placing an independent Beta prior for each pair of items on the respective comparison outcome probability.


For an example, run
```
python bradley_terry.py 
```
.

### Looking at the behavior of the model
 
The program `test_bradley_terry.py` can be run to look at the results of the Bradley-Terry model in an artificial test scenario. In this scenario, a "win probability" is selected uniformly at random for each of `num_items` items. These items then participate in `num_pairs` pairwise comparisons/competitions. 

For each pairwise comparison between two items, the program selects each item's "score" for that specific comparison from a Gaussian centered on
that item's win probability. The item in the pair with the higher score wins.

For example, if item A has "true" win probability 0.4 and B has 0.7,
the Gaussian sampling could lead to A's score being 0.45 and B's 0.65,
in which case B wins this particular competition (consistent with its higher
true win probability), or we could get A's score being 0.55 and B's 0.32,
resulting in an "upset" where the overall stronger competitor lost.

To run:

```
python test_bradley_terry.py
```

Or more generally:

```
test_bradley_terry.py <options>

Run Bradley-Terry model test and save output to a PDF.

optional arguments:
  -h, --help            show this help message and exit
  --num_items NUM_ITEMS
                        Number of items
  --num_pairs NUM_PAIRS
                        Number of pairwise comparisons
  --min_comparisons MIN_COMPARISONS
                        Minimum pairwise comparisons required 
                        to include an item
  --variance VARIANCE   Variance for Gaussian distribution
  --output OUTPUT       Output PDF filename
  --regression          Include regression line and confidence interval
  						    (defaults to True)
  --verbose             Include verbose output (defaults to False)
  --debug               Include debug output (defaults to False)
```

In the PDF that gets created, the first plot, Simulated Win Proportion vs 'True' Win Probability,
 shows how much noise has been added in the simulated competitions by using variance = 0.1

The second plot, Bradley-Terry Parameters vs Simulated Win Proportions,
 shows how well Bradley-Terry has recovered the observed (with noise) item-level proportions of wins.

The third plot, Bradley-Terry Parameters vs Simulated Win Proportions,
 shows how well Bradley-Terry has recovered the underlying ground truth for items' win probabilities, even with noise added.



