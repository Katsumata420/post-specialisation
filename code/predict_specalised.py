"""
Predict specialized vectors with the trained model.
"""
import argparse
import os
from typing import Dict, List, Set

import keras
import numpy as np
import tqdm
from sklearn.preprocessing import normalize


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--w2v-data", required=True)
    parser.add_argument("--seen-words-w2v-path", required=True)
    parser.add_argument("--output-file", required=True)
    return parser.parse_args()


def load_w2v(file_path: str) -> Dict[str, np.ndarray]:
    w2v = {}
    with open(file_path) as i_f:
        for line in i_f:
            word, vector = line.strip().split(" ", 1)
            vector = np.array(vector.split(), dtype="float32")
            norm = np.linalg.norm(vector)
            w2v[word] = vector / norm

    print(f"{len(w2v)} vectors loaded from {file_path}.")
    return w2v


def predict_and_write_specialised_vector(
    model: keras.Sequential, w2v_data: Dict[str, np.ndarray], seen_w2v: Dict[str, np.ndarray], output_file: str
) -> None:
    """
    Predict specialised vecotrs with loaded model and write vectors to output_file.
    """
    def extract_pre_specialised_words(file_path: str) -> Set[str]:
        words = set() 
        with open(file_path) as i_f:
            for line in i_f:
                word, _ = line.strip().split(" ", 1)
                words.add(word)
        return words

    is_resume = os.path.isfile(output_file)
    print(f"{output_file} already exist. Resume to predict specialised vectors.")

    w2v_vocab = set(w2v_data.keys())
    pre_specialised_words = set()
    if is_resume:
        pre_specialised_words = extract_pre_specialised_words(output_file)
    predict_target_words = w2v_vocab - pre_specialised_words
    print(f"Predict {len(predict_target_words)} words.")

    def predict_vector(word: str) -> List[List[float]]:
        if word in seen_w2v:
            seen_vector = normalize(np.array(seen_w2v[word], dtype="float32").reshape(1, -1), norm="l2", axis=1)
            seen_vector = np.ndarray.tolist(seen_vector)
            return seen_vector
        else:
            input_vector = np.asarray([w2v_data[word]])
            predict_vector = model.predict(input_vector, verbose=0)[0]  # unsqueeze for batch
            normalized_predict_vector = normalize(predict_vector.reshape(1, -1), norm="l2", axis=1)
            normalized_predict_vector = np.ndarray.tolist(normalized_predict_vector)
            return normalized_predict_vector

    if is_resume:
        target_fp = open(output_file, "a")
    else:
        target_fp = open(output_file, "w")
    for word in tqdm.tqdm(predict_target_words):
        encode_vector = predict_vector(word)
        encode_vector = " ".join([str(v) for v in encode_vector[0]])
        output_line = f"{word} {encode_vector}\n"
        target_fp.write(output_line)
    target_fp.close()


def main():
    args = get_args()
    model = keras.models.load_model(args.model_path, compile=False)
    w2v_data = load_w2v(args.w2v_data)
    seen_w2v = load_w2v(args.seen_words_w2v_path)

    predict_and_write_specialised_vector(model, w2v_data, seen_w2v, args.output_file)

if __name__ == "__main__":
    main()
