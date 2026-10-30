"""
taxonomy.py - Taxonomic annotation parser and abundance rollup aggregator.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


RANK_PREFIXES = {
    "domain": ("d__", "k__"),
    "phylum": ("p__",),
    "class": ("c__",),
    "order": ("o__",),
    "family": ("f__",),
    "genus": ("g__",),
    "species": ("s__",),
}


def parse_taxonomy_line(tax_str: str) -> Dict[str, str]:
    """
    Parse semicolon-delimited SILVA or UNITE taxonomic string into ranks.
    Example: 'd__Bacteria;p__Acidobacteriota;c__Acidobacteriae;...'
    """
    ranks = {
        "domain": "Unclassified",
        "phylum": "Unclassified",
        "class": "Unclassified",
        "order": "Unclassified",
        "family": "Unclassified",
        "genus": "Unclassified",
        "species": "Unclassified",
    }
    if not isinstance(tax_str, str) or not tax_str.strip():
        return ranks

    tokens = [t.strip() for t in tax_str.split(";")]
    for token in tokens:
        for rank_name, prefixes in RANK_PREFIXES.items():
            for pfx in prefixes:
                if token.startswith(pfx):
                    val = token[len(pfx):].strip()
                    if val and val != "uncultured" and val != "unidentified":
                        ranks[rank_name] = val
                    break
    return ranks


def rollup_taxonomy(
    counts: np.ndarray,
    feature_ids: List[str],
    taxonomy_df: pd.DataFrame,
    rank: str = "phylum"
) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate ASV/OTU counts by specified taxonomic rank.

    Args:
        counts: (n_samples, n_features) ndarray
        feature_ids: list of ASV IDs corresponding to columns of counts
        taxonomy_df: DataFrame containing 'asv_id' and 'taxonomy' (or rank column)
        rank: taxonomic level to group by (e.g., 'phylum', 'genus')

    Returns:
        (aggregated_counts, group_names)
    """
    tax_dict = {}
    if rank in taxonomy_df.columns:
        tax_dict = dict(zip(taxonomy_df["asv_id"], taxonomy_df[rank]))
    elif "taxonomy" in taxonomy_df.columns:
        for _, row in taxonomy_df.iterrows():
            parsed = parse_taxonomy_line(str(row["taxonomy"]))
            tax_dict[row["asv_id"]] = parsed.get(rank, "Unclassified")

    feature_groups = [tax_dict.get(fid, "Unclassified") for fid in feature_ids]
    unique_groups = sorted(list(set(feature_groups)))

    agg_matrix = np.zeros((counts.shape[0], len(unique_groups)), dtype=np.int64)
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}

    for col_idx, g in enumerate(feature_groups):
        agg_idx = group_to_idx[g]
        agg_matrix[:, agg_idx] += counts[:, col_idx]

    return agg_matrix, unique_groups
