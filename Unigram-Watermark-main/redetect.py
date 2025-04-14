import argparse
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, LlamaTokenizer
from gptwm import GPTWatermarkDetector
from statistics import NormalDist
from math import sqrt
import scipy.stats


def main(args):
    with open(args.input_file, 'r') as f:
        data = [json.loads(x) for x in f.read().strip().split("\n")]
    if 'llama' in args.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name, torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, torch_dtype=torch.float16)

    vocab_size = 50272 if "opt" in args.model_name else tokenizer.vocab_size

    detector = GPTWatermarkDetector(fraction=args.fraction,
                                    strength=args.strength,
                                    vocab_size=vocab_size,
                                    watermark_key=args.wm_key)

    z_score_list = []
    with open(args.input_file.replace('.jsonl', '_watermarked.jsonl'), 'w') as f:
        for idx, cur_data in tqdm(enumerate(data), total=len(data)):
            # processed = cur_data['Unwatermarked Output Post-Processing']
            processed = cur_data['Watermarked Output Post-Processing']
            gen_tokens = tokenizer(processed, add_special_tokens=False)["input_ids"]
            if len(gen_tokens) >= args.test_min_tokens:
                z_score_list.append(detector.detect(gen_tokens))
            else:
                print(f"Warning: sequence {idx} is too short to test.")

            num_green = [0,0] # first is for regular, second is unique tokens only
            # {id: decoded string, whether greenlisted, # occurrences}; may be useful to sort in post
            # must use id for key (not decoded string) because decoding collisions may occur (e.g. Llama 2 uses SentencePiece tokenizer)
            are_tokens_green = {}
            for i in gen_tokens:
                decoded = tokenizer.decode(i)
                is_green = int(detector.green_list_mask[i].item())
                if i not in are_tokens_green:
                    are_tokens_green[i] = [decoded,is_green,1]
                    num_green[1] += is_green
                else:
                    are_tokens_green[i][2] += 1
                num_green[0] += is_green

            gen_length = [len(gen_tokens), len(are_tokens_green)] # first is for regular, second is unique tokens only
            too_short = False
            if gen_length[0] < args.test_min_tokens:
                print(f"Warning: generation {idx} is too short to test.")
                too_short = True

            # same as detector.detect(gen_tokens), detector.unidetect(gen_tokens); using regular detector and "unique" detector
            z_score = [detector._z_score(num_green[0], gen_length[0], args.fraction),
                        detector._z_score(num_green[1], gen_length[1], args.fraction)]

            p_value = [scipy.stats.norm.sf(z_score[0]),scipy.stats.norm.sf(z_score[1])]

            # desired/theoretical FPR (Type-I error rate)
            # want to match this with the FPR that the regular detection threshold corresponds to
            alpha = 0.01 # can't be 0 because can't take invnorm of 1
            unique_threshold = sqrt(1-(gen_length[1]-1)/(vocab_size-1))*NormalDist().inv_cdf(1-alpha) # equation 9 from Appendix E.2, virtually constant
            wm_pred = [1 if z_score[0] > args.threshold else 0,
                    1 if z_score[1] > unique_threshold else 0]

            save_dict = {
                "time_completed": 0,
                "prefix": cur_data['Prompt'],
                "gold_completion": "",
                "gen_completion": processed,
                "too_short": too_short,
                "z-score": z_score,
                "p-value": p_value,
                "wm_pred": wm_pred,
                "gen_length": gen_length,
                "num_green": num_green,
                "are_tokens_green": are_tokens_green
            }

            json.dump(save_dict, f)
            f.write('\n')

    print('Finished!')
    # drive.flush_and_unmount()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    dataset = "Adaptive" # "OpenGen"
    parser.add_argument("--dataset", type=str, default=dataset)
    # parser.add_argument("--model_name", type=str, default="facebook/opt-125m")
    parser.add_argument("--model_name", type=str, default="facebook/opt-1.3b") # baffo32/decapoda-research-llama-7B-hf
    parser.add_argument("--strength", type=float, default=2.0) # aka delta, default was 2.0, set to 0 for unwatermarked model (doesn't actually matter here)
    parser.add_argument("--fraction", type=float, default=0.5) # aka gamma
    parser.add_argument("--threshold", type=float, default=6.0) # aka tau, but NOT the same as tau calculated for unique detector
    parser.add_argument("--wm_key", type=int, default=0)
    parser.add_argument("--input_file", type=str, default=f"./data/{dataset}/to_redetect.jsonl") #
    parser.add_argument("--test_min_tokens", type=int, default=200)
    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1") # https://stackoverflow.com/questions/48796169/how-to-fix-ipykernel-launcher-py-error-unrecognized-arguments-in-jupyter

    args = parser.parse_args()

    main(args)