import choix
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import choix

###################################################################
# Fitting the Bradley-Terry model to the specified set of items
# with the data consisting of (itemA,itemB) pairs where the first
# value is always the "winner" for that comparison.
# Returns dictionary of parameters ("strengths") for the items
###################################################################

def run_bradley_terry(pairs, alpha=0.1, verbose=False):

    # Create a sorted list of unique items
    items = sorted(set(item for pair in pairs for item in pair))

    # Create a dictionary to map each item to its index
    item_to_index = {item: idx for idx, item in enumerate(items)}

    # Convert original pairs to pairs of those indexes
    index_pairs = [(item_to_index[a], item_to_index[b]) for a, b in pairs]

    if verbose:    
        print(f"Running Bradley_Terry on {len(pairs)} pairs from {len(items)} items")
        print(f"Items: {items}")
        print(f"Pairs: {pairs}")

    # Fit the model
    params  = choix.ilsr_pairwise(len(items), index_pairs, alpha=0.01)
    result  = dict(zip(items,params))
    
    if verbose:
        print(f"Resulting Bradley-Terry parameters: {result}")
        
    return result

if __name__ == "__main__":

    print("Using hard-wired test data")

    pairs = [('item_1', 'item_0'), ('item_3', 'item_4'), ('item_3', 'item_4'), ('item_1', 'item_3'), ('item_1', 'item_4'), ('item_1', 'item_2'), ('item_3', 'item_0'), ('item_3', 'item_4'), ('item_3', 'item_2'), ('item_1', 'item_0'), ('item_1', 'item_2'), ('item_1', 'item_2'), ('item_2', 'item_0'), ('item_1', 'item_2'), ('item_2', 'item_0'), ('item_1', 'item_4'), ('item_3', 'item_2'), ('item_1', 'item_3'), ('item_3', 'item_4'), ('item_1', 'item_0'), ('item_3', 'item_4'), ('item_1', 'item_3'), ('item_2', 'item_4'), ('item_0', 'item_4'), ('item_1', 'item_3'), ('item_1', 'item_3'), ('item_3', 'item_0'), ('item_3', 'item_4'), ('item_1', 'item_2'), ('item_1', 'item_4'), ('item_3', 'item_4'), ('item_1', 'item_3'), ('item_4', 'item_2'), ('item_2', 'item_0'), ('item_3', 'item_0'), ('item_2', 'item_0'), ('item_2', 'item_4'), ('item_1', 'item_0'), ('item_1', 'item_0'), ('item_2', 'item_0'), ('item_3', 'item_0'), ('item_1', 'item_4'), ('item_1', 'item_2'), ('item_3', 'item_2'), ('item_3', 'item_1'), ('item_0', 'item_2'), ('item_2', 'item_4'), ('item_3', 'item_4'), ('item_1', 'item_2'), ('item_1', 'item_0'), ('item_3', 'item_2'), ('item_2', 'item_0'), ('item_1', 'item_0'), ('item_3', 'item_0'), ('item_1', 'item_2'), ('item_1', 'item_2'), ('item_0', 'item_2'), ('item_2', 'item_0'), ('item_1', 'item_4'), ('item_3', 'item_0'), ('item_2', 'item_0'), ('item_3', 'item_0'), ('item_1', 'item_3'), ('item_1', 'item_4'), ('item_1', 'item_3'), ('item_1', 'item_4'), ('item_1', 'item_4'), ('item_2', 'item_4'), ('item_1', 'item_2'), ('item_2', 'item_0'), ('item_3', 'item_4'), ('item_3', 'item_0'), ('item_3', 'item_4'), ('item_2', 'item_0'), ('item_4', 'item_0'), ('item_3', 'item_0'), ('item_3', 'item_0'), ('item_1', 'item_3'), ('item_1', 'item_2'), ('item_0', 'item_4'), ('item_2', 'item_0'), ('item_1', 'item_2'), ('item_2', 'item_3'), ('item_1', 'item_3'), ('item_3', 'item_2'), ('item_0', 'item_4'), ('item_1', 'item_2'), ('item_1', 'item_4'), ('item_1', 'item_0'), ('item_3', 'item_0'), ('item_1', 'item_0'), ('item_1', 'item_2'), ('item_3', 'item_0'), ('item_1', 'item_3'), ('item_1', 'item_3'), ('item_1', 'item_0'), ('item_3', 'item_0'), ('item_2', 'item_4'), ('item_3', 'item_4'), ('item_1', 'item_4'), ('item_1', 'item_2'), ('item_1', 'item_4'), ('item_1', 'item_0'), ('item_3', 'item_4'), ('item_1', 'item_2'), ('item_1', 'item_4'), ('item_1', 'item_0'), ('item_1', 'item_2'), ('item_1', 'item_3'), ('item_1', 'item_2'), ('item_3', 'item_4'), ('item_2', 'item_0'), ('item_0', 'item_4'), ('item_3', 'item_0'), ('item_2', 'item_4'), ('item_2', 'item_4'), ('item_4', 'item_0'), ('item_1', 'item_3'), ('item_1', 'item_2'), ('item_3', 'item_0'), ('item_3', 'item_2'), ('item_1', 'item_0'), ('item_2', 'item_4'), ('item_1', 'item_3'), ('item_2', 'item_0'), ('item_1', 'item_4'), ('item_2', 'item_4'), ('item_2', 'item_4'), ('item_1', 'item_0'), ('item_2', 'item_4'), ('item_1', 'item_0'), ('item_2', 'item_4'), ('item_1', 'item_0'), ('item_2', 'item_3'), ('item_1', 'item_4'), ('item_1', 'item_2'), ('item_1', 'item_4'), ('item_1', 'item_2'), ('item_0', 'item_4'), ('item_2', 'item_0'), ('item_3', 'item_0'), ('item_2', 'item_4'), ('item_2', 'item_0'), ('item_2', 'item_4'), ('item_2', 'item_0'), ('item_1', 'item_4'), ('item_3', 'item_1'), ('item_1', 'item_2'), ('item_2', 'item_4'), ('item_2', 'item_0'), ('item_0', 'item_4'), ('item_2', 'item_0'), ('item_2', 'item_0'), ('item_1', 'item_0'), ('item_2', 'item_4'), ('item_1', 'item_4'), ('item_2', 'item_4'), ('item_1', 'item_3'), ('item_3', 'item_2'), ('item_0', 'item_4'), ('item_3', 'item_4'), ('item_1', 'item_4'), ('item_4', 'item_0'), ('item_3', 'item_2'), ('item_3', 'item_0'), ('item_1', 'item_0'), ('item_2', 'item_3'), ('item_3', 'item_2'), ('item_3', 'item_2'), ('item_4', 'item_2'), ('item_3', 'item_2'), ('item_2', 'item_0'), ('item_1', 'item_2'), ('item_3', 'item_0'), ('item_1', 'item_0'), ('item_3', 'item_0'), ('item_1', 'item_0'), ('item_1', 'item_3'), ('item_2', 'item_0'), ('item_1', 'item_2'), ('item_3', 'item_2'), ('item_1', 'item_3'), ('item_2', 'item_3'), ('item_1', 'item_0'), ('item_1', 'item_0'), ('item_1', 'item_3'), ('item_1', 'item_2'), ('item_3', 'item_2'), ('item_3', 'item_0'), ('item_1', 'item_4'), ('item_1', 'item_2'), ('item_4', 'item_3'), ('item_1', 'item_0'), ('item_0', 'item_4'), ('item_1', 'item_0'), ('item_1', 'item_0'), ('item_3', 'item_4'), ('item_1', 'item_4'), ('item_1', 'item_4'), ('item_3', 'item_2')]

    result = run_bradley_terry(pairs, verbose=True)
    
    print("\nExpected Bradley-Terry parameters: {'item_0': -2.1972240202452302, 'item_1': 3.7595311437859364, 'item_2': -0.1483223128040927, 'item_3': 1.2406835264046785, 'item_4': -2.6546683371412922}")
    
