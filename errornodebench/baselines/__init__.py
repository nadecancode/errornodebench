"""Prompt-based memory baselines from the Auto-Dreamer paper (arXiv 2605.20616).

These are inference-only consolidation strategies adapted from prior work so
they can be benchmarked alongside the Fresh / Static-Group / Cumulative arms
in the Interference scenario. Implementations focus on the *distinctive
operation* of each baseline (what the prompt asks the model to do) rather
than full reproduction of the original codebases.
"""
