#!/usr/bin/python
"""
CS221 Homework 6: Bayesian Networks
"""

from util import (
    BayesianNetwork, BayesianNode, init_zero_conditional_probability_tables,
    normalize_counts, load_annotation_csv, plot_annotator_cpts, plot_label_cpt
)
from typing import Dict, List, Any, Optional, Tuple
from itertools import product
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict

############################################################
# Problem 2a: Converting Phylogenetic Tree to Bayesian Network

def initialize_phylogenetic_tree(mutation_rate: float, genome_length: int=1) -> BayesianNetwork:
    """
    Initialize the phylogenetic tree as a Bayesian network using the BayesianNode and
    Bayesian Network classes in util.py.
    """
    if not (0.0 <= mutation_rate <= 1.0):
        raise ValueError("mutation_rate must be in [0, 1].")
    # BEGIN_YOUR_CODE (our solution is 19 line(s) of code, but don't worry if you deviate from this)
    # nodes = []
    # network = BayesianNetwork(nodes=nodes)
    domain = ["A", "C", "T", "G"]
    thomas_bayus = BayesianNode(name="Thomas bayus", domain=domain, parents=[], conditional_prob_table=np.array([[0.25, 0.25, 0.25, 0.25]]))
    humblus_studentus = BayesianNode(name="Humblus studentus", domain=domain, parents=[thomas_bayus], 
                                     conditional_prob_table=np.array([[1 - mutation_rate, mutation_rate / 3, mutation_rate / 3, mutation_rate / 3],
                                                                      [mutation_rate / 3, 1 - mutation_rate, mutation_rate / 3, mutation_rate / 3],
                                                                      [mutation_rate / 3, mutation_rate / 3, 1 - mutation_rate, mutation_rate / 3],
                                                                      [mutation_rate / 3, mutation_rate / 3, mutation_rate / 3, 1 - mutation_rate]]))
    aryamus_bayus = BayesianNode(name="Aryamus bayus", domain=domain, parents=[thomas_bayus], 
                                     conditional_prob_table=np.array([[1 - mutation_rate, mutation_rate / 3, mutation_rate / 3, mutation_rate / 3],
                                                                      [mutation_rate / 3, 1 - mutation_rate, mutation_rate / 3, mutation_rate / 3],
                                                                      [mutation_rate / 3, mutation_rate / 3, 1 - mutation_rate, mutation_rate / 3],
                                                                      [mutation_rate / 3, mutation_rate / 3, mutation_rate / 3, 1 - mutation_rate]]))
    kenius_bayus = BayesianNode(name="Kenius bayus", domain=domain, parents=[aryamus_bayus], 
                                     conditional_prob_table=np.array([[1 - mutation_rate, mutation_rate / 3, mutation_rate / 3, mutation_rate / 3],
                                                                      [mutation_rate / 3, 1 - mutation_rate, mutation_rate / 3, mutation_rate / 3],
                                                                      [mutation_rate / 3, mutation_rate / 3, 1 - mutation_rate, mutation_rate / 3],
                                                                      [mutation_rate / 3, mutation_rate / 3, mutation_rate / 3, 1 - mutation_rate]]))
    # END_YOUR_CODE
    network = BayesianNetwork([aryamus_bayus, humblus_studentus, thomas_bayus, kenius_bayus], batch_size=genome_length)
    return network

############################################################
# Problem 2b: Sampling from Bayesian Networks

def forward_sampling(network: BayesianNetwork) -> Dict[str, List[str]]:
    """
    Sample a single observation from the given Bayesian network.

    Use the topological ordering of variables in network.order and sample each
    variable according to its conditional probability distribution given the values
    of its parents (which have already been sampled).

    Args:
        network: A BayesianNetwork object containing nodes to sample from

    Returns:
        A dictionary mapping variable names to their sampled values
    """
    samples: Dict[str, List[str]] = {node.name: [] for node in network.order}

    for idx in range(network.batch_size):
        assignment: Dict[str, str] = {}
        # BEGIN_YOUR_CODE (our solution is 10 line(s) of code, but don't worry if you deviate from this)
        for node in network.order:
            # Look at the domain, then compute corresponding probs
            domain = node.domain
            # [print(d) for d in domain]
            probs = [node.get_probability(d, parent_values=assignment) for d in domain]
            # print(probs)
            if not (isinstance(probs[0], float) or isinstance(probs[0], np.float64) or probs[0].size == 1):
                probs = probs[0]
            # print(probs)
            # Now, sample from domain based on weights in probs
            # print(probs, sum(probs))
            assert sum(probs) == 1.0, "Probs should sum to 1.0"         # Debugging. Do this to sanity check our approach
            choices = random.choices(domain, weights=probs, k=1)
            choice = choices[0]

            assignment[node.name] = choice               # Choices w/ k=1 returns a len 1 list
            samples[node.name].append(choice)
        # END_YOUR_CODE

    return samples

############################################################
# Problem 2c: Computing Joint Probability

def compute_joint_probability(
    network: BayesianNetwork,
    assignment: Dict[str, List[str]],
    batch_indices: Optional[List[int]] = None
) -> float:
    """
    Compute the joint probability of a given assignment to all variables in the network.

    Args:
        network: A BayesianNetwork object
        assignment: Dictionary mapping variable names to their assigned values
        batch_indices: Optional list of batch indices to compute the joint probability for
            (default is [0, 1, ..., batch_size-1], so all the indices)

    Returns:
        The joint probability as a float
    """
    # BEGIN_YOUR_CODE (our solution is 16 line(s) of code, but don't worry if you deviate from this)
    if batch_indices is None:
        batch_indices = range(network.batch_size)

    prob = 1.0
    for idx in batch_indices:
        parent_values: Dict[str, str] = {}
        for node in network.order:
            local_prob = node.get_probability(assignment[node.name][idx], parent_values=parent_values)
            # Weirdly, root nodes (no parents) return a [val] rather than just value. I suspect this is weird business bc of the cpt formatting?
            if isinstance(local_prob, np.ndarray):
                local_prob = local_prob[0]
            prob *= local_prob
            parent_values[node.name] = assignment[node.name][idx]
    return prob
    # END_YOUR_CODE

# ############################################################
# # Problem 2d: Test forward sampling

def test_forward_sampling():
    np.random.seed(123)
    random.seed(123)
    # BEGIN_YOUR_CODE (our solution is 3 line(s) of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE
    print(sample)
    print(f"{joint_probability:.10%}")

# Uncomment to test forward sampling
# test_forward_sampling()

############################################################
# Problem 2e: Rejection Sampling

def rejection_sampling(
    network: BayesianNetwork,
    target_variable: str,
    conditioned_on_assignments: Dict[str, List[str]],
    num_samples: int
) -> Dict[Any, float]:
    """
    Use rejection sampling to estimate the likelihoods for each outcome in the 
    target variable conditioned on the given assignments (a dictionary mapping
    variable names to their assigned values), i.e. P(target_variable | conditioned_on_assignments).

    Args:
        network: A BayesianNetwork object
        target_variable: The name of the variable to estimate the likelihoods for
        conditioned_on_assignments: A dictionary mapping variable names to their assigned values
        num_samples: The number of samples to draw

    Returns:
        A dictionary mapping outcomes to their likelihoods, conditioned on the given assignments.
    """
    # BEGIN_YOUR_CODE (our solution is 14 line(s) of code, but don't worry if you deviate from this)
    unnormalized: Dict[str, int] = {}

    for _ in range(num_samples):
        sample = forward_sampling(network)
        # Check if sample conforms with conditioning
        valid = True
        for cond_name, cond_form in conditioned_on_assignments.items():
            if sample[cond_name] != cond_form:
                valid = False
                break

        if valid:
            hashable_key = "".join(sample[target_variable])
            if hashable_key not in unnormalized:
                unnormalized[hashable_key] = 0
            unnormalized[hashable_key] += 1
    
    dist_mass = sum(unnormalized.values())
    normalized: Dict[List[str], float]= {k: v / dist_mass for k, v in unnormalized.items()}
    return normalized
    # END_YOUR_CODE

############################################################
# Problem 2f: Gibbs Sampling

def gibbs_sampling(
    network: BayesianNetwork,
    target_variable: str,
    conditioned_on_assignments: Dict[str, List[str]],
    num_iterations: int,
    initial_state: Dict[str, List[str]] = None
) -> Dict[Any, float]:
    """
    Estimate P(target_variable | conditioned_on_assignments) via Gibbs sampling.

    - Initialization: random ancestral sample (forward_sampling)
    - Use compute_joint_probability to score single-variable proposals
      (simple and robust for small networks)
    - Record the Thomas bayus genome after each full sweep
    """
    # random initialization, then set evidence variables
    state = forward_sampling(network) if initial_state is None else initial_state
    for evidence_var, evidence_val in conditioned_on_assignments.items():
        state[evidence_var] = evidence_val

    # resample all non-evidence vars each sweep
    resample_nodes = [n for n in network.order if n.name not in conditioned_on_assignments]
    counts = defaultdict(int)

    for _ in range(num_iterations):
        # BEGIN_YOUR_CODE (our solution is 12 line(s) of code, but don't worry if you deviate from this)
        for node in resample_nodes:
            # Sample something new for state[node.name] and update counts
            # How do we sample something new? 
            # In forward sampling, we pick the top thing from the domain.
            # In this case, we should do something similar but weighting elements in 
            # the domain by the join probability.
            for i in range(network.batch_size):
                domain = node.domain
                probs = []
                for d in domain:
                    new_state = state.copy()
                    new_state[node.name][i] = d
                    prob = compute_joint_probability(network, new_state, batch_indices=[i])
                    probs.append(prob)
                choice = random.choices(domain, probs, k=1)[0]
                state[node.name][i] = choice
            counts["".join(state[node.name])] += 1
        # END_YOUR_CODE
    total_samples = sum(counts.values())
    return {val: counts[val] / total_samples for val in counts.keys()}

def test_gibbs_vs_rejection(
    num_steps: int=10000,
    seed: int=0,
    mutation_rate: float=0.1,
    genome_length: int=4,
):
    def exact_inference():
        same = 1.0 - mutation_rate
        diff = mutation_rate / 3.0
        p_a = same * same + 3 * diff * diff
        p_not_a = diff * same + same * diff + 2 * diff * diff
        expected = (p_a ** 3) * p_not_a
        return expected
    np.random.seed(seed)
    random.seed(seed)
    network = initialize_phylogenetic_tree(mutation_rate=mutation_rate, genome_length=genome_length)
    kenius_bayus_genome = ['A'] * genome_length

    vals = gibbs_sampling(
        network, 'Thomas bayus', {'Kenius bayus': kenius_bayus_genome}, num_iterations=num_steps)
    gibbs_p_aaac = vals.get(('A', 'A', 'A', 'C'), 0.0)

    rejection_vals = rejection_sampling(
        network, 'Thomas bayus', {'Kenius bayus': kenius_bayus_genome}, num_samples=num_steps)
    rejection_p_aaac = rejection_vals.get(('A', 'A', 'A', 'C'), 0.0)

    print(f"{'Gibbs':>10} {gibbs_p_aaac:.4f}")
    print(f"{'Rejection':>10} {rejection_p_aaac:.4f}")
    print(f"{'Exact':>10} {exact_inference():.4f}")

# Uncomment to test Gibbs vs. rejection sampling
# test_gibbs_vs_rejection(num_steps=100, mutation_rate=0.1, genome_length=4)
# test_gibbs_vs_rejection(num_steps=10000, mutation_rate=0.1, genome_length=4)

############################################################
# Problem 3d: Bayesian network for annotators

def bayesian_network_for_annotators(num_annotators: int, dataset_size: int=1) -> BayesianNetwork:
    """
    Return the Bayesian network for the annotators.
    """
    # BEGIN_YOUR_CODE (our solution is 14 line(s) of code, but don't worry if you deviate from this)
    domain = ["good", "bad"]
    # Assuming root is uniformly distributed. 
    root = BayesianNode(name="Y", domain=domain, parents=[], conditional_prob_table=np.array([[0.5, 0.5]]))
    nodes = [root]
    for i in range(num_annotators):
        annotator = BayesianNode(name=f"A_{i}", domain=domain, parents=[root], conditional_prob_table=np.array([[0.6, 0.4], [0.2, 0.8]]))
        nodes.append(annotator)
    network = BayesianNetwork(nodes, batch_size=dataset_size)
    return network
    # END_YOUR_CODE

############################################################
# Problem 3e: Maximum likelihood estimation

def accumulate_assignment(
    counts: Dict[str, np.ndarray],
    network: BayesianNetwork,
    assignment: Dict[str, str],
    weight: float = 1.0,
    batch_indices: Optional[List[int]] = None,
) -> None:
    """
    Add weighted counts for a fully or partially observed assignment.
    """
    batch_size = len(list(assignment.values())[0])
    for i in range(batch_size) if batch_indices is None else range(len(batch_indices)):
        idx = batch_indices[i] if batch_indices is not None else i
        assignment_i = {k: v[idx] for k, v in assignment.items()}           # This should be v[idx] not v[i] I think...
        for node in network.nodes:
            # BEGIN_YOUR_CODE (our solution is 7 line(s) of code, but don't worry if you deviate from this)
            if len(node.parents) == 0:
                k = node.domain.index(assignment_i[node.name])
                counts[node.name][0, k] += weight
            else:
                parent_indices = node.parent_assignment_indices(assignment_i)
                for parent_i, parent in enumerate(node.parents):
                    j = parent_indices[parent_i]
                    k = node.domain.index(assignment_i[node.name])
                    counts[node.name][j, k] += weight
            # END_YOUR_CODE

def mle_estimation(network: BayesianNetwork, data: List[Dict[str, List[str]]], lambda_param: float = 1.0) -> BayesianNetwork:
    """
    Return the Bayesian network with the parameters estimated by MLE.
    """
    # BEGIN_YOUR_CODE (our solution is 7 line(s) of code, but don't worry if you deviate from this)
    # todo: use accumulate_assignment to initialize the cpt tables (params) for each network
    counts = init_zero_conditional_probability_tables(network)
    for cpt in counts.values():
        cpt += lambda_param
    for assignment in data:
        accumulate_assignment(counts, network, assignment)
    normalize_counts(network, counts)
    return network
    # END_YOUR_CODE

############################################################
# Problem 3f: Maximum likelihood estimation for annotators

def mle_estimation_for_annotators(data: List[Dict[str, List[str]]]) -> BayesianNetwork:
    """
    Return the Bayesian network with the parameters estimated by MLE for the annotators.
    """
    # BEGIN_YOUR_CODE (our solution is 2 line(s) of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

def test_mle_estimation_for_annotators():
    data = load_annotation_csv('data/annotations.csv', include_labels=True)
    trained = mle_estimation_for_annotators(data)
    plot_annotator_cpts(trained, "plots/annotators.png")

# test_mle_estimation_for_annotators()

############################################################
# Problem 4a: Expectation step

def e_step(
    network: BayesianNetwork, data: List[Dict[str, List[str]]]
) -> Tuple[List[Dict[str, List[str]]], List[float], List[List[int]]]:
    """
    Create the dataset of fully-observed weighted observations given some hidden variables, for the EM algorithm.
    """
    # BEGIN_YOUR_CODE (our solution is 33 line(s) of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 4b: Maximization step

def m_step(
    network: BayesianNetwork,
    all_completions: List[Dict[str, str]],
    all_weights: List[float],
    all_indices: List[int],
) -> BayesianNetwork:
    """
    Update the CPTs of the Bayesian network using expected counts.
    """
    # BEGIN_YOUR_CODE (our solution is 5 line(s) of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 4c: Expectation maximization

def em_learn(network: BayesianNetwork, data: List[Dict[str, str]], num_iterations: int) -> BayesianNetwork:
    """
    Run the EM algorithm for a given number of iterations.
    """
    # BEGIN_YOUR_CODE (our solution is 4 line(s) of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

def test_em_learn():
    network = bayesian_network_for_annotators(num_annotators=3, dataset_size=100)
    data = load_annotation_csv('data/annotations.csv', include_labels=False)
    trained = em_learn(network, data, num_iterations=100)
    plot_annotator_cpts(trained, "plots/annotators_em.png")
    plot_label_cpt(trained, "plots/labels_em.png")

# Uncomment to test EM learning
# test_em_learn()
