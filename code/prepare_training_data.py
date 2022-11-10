"""
Extract seen words and vecotrs from original and specialized w2v files.
"""
import argparse
import logging
import os
import sys
from typing import Dict, Tuple

import numpy as np


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler_format = logging.Formatter("%(asctime)s : %(levelname)s - %(filename)s - %(message)s")
stream_hander = logging.StreamHandler(stream=sys.stdout)
stream_hander.setLevel(logging.DEBUG)
stream_hander.setFormatter(handler_format)
logger.addHandler(stream_hander)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, help="original w2v file path.")
    parser.add_argument("--specialized", required=True, help="specialized w2v file path.")
    parser.add_argument("--output-dir", required=True, help="output dir in which training data is saved.")
    parser.add_argument(
        "--is-first-line-ignore", action="store_true", help="if true, ignore first line in loading w2v file."
    )
    return parser.parse_args()


def load_w2v(file_path: str, first_line_ignore: bool = False) -> Dict[str, np.ndarray]:
    """Load w2v file from text format.

        Args:
          file_path (str): w2v file path.
          first_line_ignore (bool): if true, ignore first line at w2v file.

        Returns:
          Dict[str, str]: key is word, value is vector.
    """
    w2v_data = {}
    with open(file_path) as i_f:
        for idx, line in enumerate(i_f):
            if idx == 0 and first_line_ignore:
                continue
            word, vector = line.strip().split(" ", 1)
            w2v_data[word] = np.fromstring(vector, dtype="float32", sep=" ")

    logger.info(f"Loaded from {file_path}. Size: {len(w2v_data)}")
    return w2v_data


def extract_seen_words(
    original_w2v: Dict[str, np.ndarray], specialized_w2v: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Extract w2v data as for seen words."""

    original_seen_w2v = {}
    specialized_seen_w2v = {}

    for vocab in original_w2v.keys():
        original_vector = original_w2v[vocab]
        specialized_vector = specialized_w2v[vocab]
        if not np.all(original_vector == specialized_vector):
            original_seen_w2v[vocab] = original_vector
            specialized_seen_w2v[vocab] = specialized_vector
    
    logger.info(f"Detected seen word size: {len(original_seen_w2v)}")
    return original_seen_w2v, specialized_seen_w2v


def save_w2v(w2v_data: Dict[str, np.ndarray], file_path: str) -> None:
    with open(file_path, "w") as o_f:
        for vocab, vector in w2v_data.items():
            vector = " ".join([str(v) for v in vector])
            output_line = f"{vocab} {vector}\n"
            o_f.write(output_line)


def main():
    args = get_args()

    logger.info("Load w2v data.")
    original_w2v = load_w2v(args.original)
    specialized_w2v = load_w2v(args.specialized)
    assert len(original_w2v) == len(specialized_w2v), "Vocab size must be equal."

    logger.info("Extract w2v data from seen words.")
    original_seen_w2v, specialized_seen_w2v = extract_seen_words(original_w2v, specialized_w2v)

    logger.info("Save w2v data of seen words.")
    if not os.path.isdir(args.output_dir): 
        os.mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, "original_seen_w2v.data")
    save_w2v(original_seen_w2v, output_path)
    output_path = os.path.join(args.output_dir, "specialized_seen_w2v.data")
    save_w2v(specialized_seen_w2v, output_path)


if __name__ == "__main__":
    main()
