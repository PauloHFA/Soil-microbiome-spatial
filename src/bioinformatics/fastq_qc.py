"""
fastq_qc.py - FASTQ quality control, dual-orientation resolution, and primer trimming.
"""

from typing import Tuple, Optional


IUPAC_MAP = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "R": "AG", "Y": "CT", "S": "GC", "W": "AT",
    "K": "GT", "M": "AC", "B": "CGT", "D": "AGT",
    "H": "ACT", "V": "ACG", "N": "ACGT"
}

REVCOMP_TRANS = str.maketrans("ACGTURYKMSWBDHVNacgturykmswbdhvn", "TGCAAYRMKWSVHDBNtgcaayrmkwsvhdbn")


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a nucleotide sequence."""
    return sequence.translate(REVCOMP_TRANS)[::-1]


def match_iupac_char(seq_char: str, primer_char: str) -> bool:
    """Check if sequence character matches degenerate IUPAC primer character."""
    allowed = IUPAC_MAP.get(primer_char.upper(), "ACGT")
    return seq_char.upper() in allowed


def match_primer(seq: str, primer: str, max_mismatches: int = 2) -> Optional[int]:
    """
    Search for primer at start of seq allowing up to max_mismatches.
    Returns prefix length if matched, else None.
    """
    p_len = len(primer)
    if len(seq) < p_len:
        return None

    mismatches = 0
    for s_char, p_char in zip(seq[:p_len], primer):
        if not match_iupac_char(s_char, p_char):
            mismatches += 1
            if mismatches > max_mismatches:
                return None
    return p_len


def detect_orientation(sequence: str, primer: str, max_mismatches: int = 2) -> str:
    """
    Determine if read is in FORWARD or REVERSE_COMPLEMENT orientation relative to primer.
    """
    if match_primer(sequence, primer, max_mismatches) is not None:
        return "FORWARD"

    rc_seq = reverse_complement(sequence)
    if match_primer(rc_seq, primer, max_mismatches) is not None:
        return "REVERSE_COMPLEMENT"

    return "UNKNOWN"


def reorient_read(sequence: str, primer: str, max_mismatches: int = 2) -> Tuple[str, str]:
    """
    Reorient read to FORWARD orientation if in REVERSE_COMPLEMENT.
    Returns (reoriented_sequence, detected_orientation).
    """
    orientation = detect_orientation(sequence, primer, max_mismatches)
    if orientation == "REVERSE_COMPLEMENT":
        return reverse_complement(sequence), orientation
    return sequence, orientation


def trim_primer(sequence: str, primer: str, max_mismatches: int = 2) -> str:
    """
    Trim primer prefix from sequence if found within max_mismatches.
    """
    p_len = match_primer(sequence, primer, max_mismatches)
    if p_len is not None:
        return sequence[p_len:]
    return sequence
