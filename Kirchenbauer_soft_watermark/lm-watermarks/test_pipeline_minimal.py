import argparse
from tqdm import tqdm
import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LogitsProcessorList, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from time import time
import gc
import signal
import sys
from functools import partial
from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

def str2bool(v): # from demo_watermark.py
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def read_file(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f.read().strip().split("\n")]

# obsolete; not recommended for frequent writes, use with large batch writes only (like in original code), flushes after with-block closes per call
def write_file(filename, data):
    with open(filename, "a") as f:
        f.write("\n".join(data) + "\n")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use gpu
    print(f"Model will run on: {device}")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, llm_int8_enable_fp32_cpu_offload=True, bnb_4bit_compute_dtype=torch.float16)
    device_map = 'auto'
    # device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 0, 'model.layers.17': 0, 'model.layers.18': 0, 'model.layers.19': 0, 'model.layers.20': 0, 'model.layers.21': 0, 'model.layers.22': 0, 'model.layers.23': 0, 'model.layers.24': 'cpu', 'model.layers.25': 'cpu', 'model.layers.26': 'cpu', 'model.layers.27': 'cpu', 'model.layers.28': 'cpu', 'model.layers.29': 'cpu', 'model.layers.30': 'cpu', 'model.layers.31': 'cpu', 'model.norm': 'cpu', 'model.rotary_emb': 'cpu', 'lm_head': 'cpu'}

    # output_file = f"{args.model_name.replace('/', '-')}_strength_{args.strength}_frac_{args.fraction}_len_{args.max_new_tokens}_"
    # new naming convention:
    output_file = "minimal,"+",".join([f'{t[1]}' for t in list(vars(args).items())[:-5]]).replace('/', '-') +",v"
    if args.avoid_same_file == 0:
        output_file = args.output_dir + output_file + '0.jsonl'
        print(output_file)
    else:
        max_dupe = -1
        for file_name in os.listdir(args.output_dir):
            if file_name.startswith(output_file):
                number = int(file_name[len(output_file):-6])
                if number > max_dupe:
                    max_dupe = number
        output_file = args.output_dir + output_file + f'{max_dupe+1}.jsonl'

    data = read_file(args.prompt_file)
    num_cur_outputs = len(read_file(output_file)) if os.path.exists(output_file) else 0

    if 'llama' in args.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name, torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, torch_dtype=torch.float16)

    # adding offload folder for weights (if needed), "auto" should primarily use gpu, 4bit quantization
    # after quantization, I didn't actually need the offload folder on disk (see my device maps below), but just keeping it here for general use
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map=device_map, offload_folder='./offload/', quantization_config=quantization_config)
    # to specify your device_map, print out quantized hf_device_map with device_map="auto" and then copy-paste it as the argument
    print(model.hf_device_map)
    # device_map="auto" (unquantized): {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 'cpu', 'model.layers.3': 'cpu', 'model.layers.4': 'cpu', 'model.layers.5': 'cpu', 'model.layers.6': 'cpu', 'model.layers.7': 'cpu', 'model.layers.8': 'cpu', 'model.layers.9': 'cpu', 'model.layers.10': 'cpu', 'model.layers.11': 'disk', 'model.layers.12': 'disk', 'model.layers.13': 'disk', 'model.layers.14': 'disk', 'model.layers.15': 'disk', 'model.layers.16': 'disk', 'model.layers.17': 'disk', 'model.layers.18': 'disk', 'model.layers.19': 'disk', 'model.layers.20': 'disk', 'model.layers.21': 'disk', 'model.layers.22': 'disk', 'model.layers.23': 'disk', 'model.layers.24': 'disk', 'model.layers.25': 'disk', 'model.layers.26': 'disk', 'model.layers.27': 'disk', 'model.layers.28': 'disk', 'model.layers.29': 'disk', 'model.layers.30': 'disk', 'model.layers.31': 'disk', 'model.norm': 'disk', 'model.rotary_emb': 'disk', 'lm_head': 'disk'}
    # device_map="auto" (quantized): {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 0, 'model.layers.17': 0, 'model.layers.18': 0, 'model.layers.19': 0, 'model.layers.20': 0, 'model.layers.21': 0, 'model.layers.22': 0, 'model.layers.23': 0, 'model.layers.24': 'cpu', 'model.layers.25': 'cpu', 'model.layers.26': 'cpu', 'model.layers.27': 'cpu', 'model.layers.28': 'cpu', 'model.layers.29': 'cpu', 'model.layers.30': 'cpu', 'model.layers.31': 'cpu', 'model.norm': 'cpu', 'model.rotary_emb': 'cpu', 'lm_head': 'cpu'}
    model.eval()

    watermark_processor = LogitsProcessorList([WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                                        gamma=args.fraction,
                                                                        delta=args.strength,
                                                                        seeding_scheme=args.seeding_scheme)])

    detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                 gamma=args.fraction, # should match original setting
                                 seeding_scheme=args.seeding_scheme, # should match original setting
                                 device=model.device, # must match the original rng device type
                                 tokenizer=tokenizer,
                                 z_threshold=args.threshold,
                                 normalizers=args.normalizers,
                                 ignore_repeated_bigrams=False)
    
    # ignores repeated bigrams
    unique_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                 gamma=args.fraction, # should match original setting
                                 seeding_scheme=args.seeding_scheme, # should match original setting
                                 device=model.device, # must match the original rng device type
                                 tokenizer=tokenizer,
                                 z_threshold=args.threshold,
                                 normalizers=args.normalizers,
                                 ignore_repeated_bigrams=True)

    initial_time = int(time())
    print('initial_time:',initial_time)
    outputs = []

    def manual_exit(signum, frame, ask=True):
        print(f'\nHandling signal {signum} ({signal.Signals(signum).name}).')
        if ask:
            signal.signal(signal.SIGINT, partial(manual_exit, ask=False))
            print('Press ctrl-C again to confirm exit.')
            return
        f.flush()
        if outputs:
            f.write("\n".join(outputs) + "\n") # changed to only open output file in append mode once with a single with-block
            f.flush() # to see outputs immediately (originally implicitly upon each with-block closing upon write_file return)
        # these seemed to solve issue where generation would take longer with more iterations and sudden termination would worsen the issue even after restarting
        gc.collect() # clear unused CPU RAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # clear unused GPU VRAM
            torch.cuda.ipc_collect() # clear unused GPU VRAM from terminated processes
        # drive.flush_and_unmount()
        sys.exit(0) # raises SystemExit exception, so with-block will close file safely

    # signal handlers for safe manual exit
    signal.signal(signal.SIGINT, manual_exit) # handle ctrl-C (if ctrl-C doesn't work then can kill terminal and restart computer)
    signal.signal(signal.SIGTERM, manual_exit)

    # single with-block instead of opening file multiple times to reduce overhead and corruption risk
    with open(output_file, "a") as f:
        for idx, cur_data in tqdm(enumerate(data), total=min(len(data),args.num_test)):
            if idx < num_cur_outputs or num_cur_outputs >= args.num_test or len(outputs) >= args.num_test:
                continue

            if "gold_completion" not in cur_data and 'targets' not in cur_data:
                gold_completion = ''
            elif "gold_completion" in cur_data:
                prefix = cur_data['prefix']
                gold_completion = cur_data['gold_completion']
            else:
                prefix = cur_data['prefix']
                gold_completion = cur_data['targets'][0]
            prefix = cur_data['prefix']

            batch = tokenizer(prefix, truncation=True, return_tensors="pt").to(model.device) # inputs should be on same device as model (accelerate handles device map)
            num_tokens = len(batch['input_ids'][0])

            # these seemed to solve issue where generation would take longer with more iterations and sudden termination would worsen the issue even after restarting
            gc.collect() # clear unused CPU RAM
            if torch.cuda.is_available():
                torch.cuda.empty_cache() # clear unused GPU VRAM
                torch.cuda.ipc_collect() # clear unused GPU VRAM from terminated processes

            with torch.inference_mode():
                generate_args = {
                    **batch,
                    'logits_processor': watermark_processor,
                    'output_scores': True,
                    'return_dict_in_generate': True,
                    'max_new_tokens': args.max_new_tokens,
                }

                if args.use_sampling:
                    generate_args.update(dict(
                        do_sample=True, 
                        top_k=0,
                        temperature=args.sampling_temp
                    ))
                else:
                    generate_args.update(dict(
                        num_beams=args.n_beams
                    ))

                torch.manual_seed(args.generation_seed) # makes outputs deterministic given same prompts...same as in extended implementation

                generation = model.generate(**generate_args) # the bulk of the computation time
                gen_text = tokenizer.batch_decode(generation['sequences'][:, num_tokens:], skip_special_tokens=True)

            if torch.cuda.is_available():
                print(f'\nGPU memory currently allocated: {100*torch.cuda.memory_allocated() / torch.cuda.memory_reserved():.2f}% ({torch.cuda.memory_allocated()/1024**2:.0f}/{torch.cuda.memory_reserved()/1024**2:.0f} MB)')

            gen_tokens = tokenizer(gen_text[0], add_special_tokens=False)["input_ids"]

            score_dict = detector.detect(gen_text[0], return_green_token_mask = True)
            score_dict_unique = unique_detector.detect(gen_text[0])

            num_green = [score_dict['num_green_tokens'],score_dict_unique['num_green_tokens']] # first is for regular, second is unique ngrams only
            # {id: decoded string, # green occurrences, # red occurrences}; may be useful to sort in post
            # unlike Unigram, same token could possibly be green in one instance but red in another
            # must use id for key (not decoded string) because decoding collisions may occur (e.g. Llama 2 uses SentencePiece tokenizer)
            are_tokens_green = {}
            for index,i in enumerate(gen_tokens):
                if index >= len(score_dict['green_token_mask']):
                    # with simple hash's context width of 1, for some reason len(gen_tokens) is always 1 more than # tokens scored???
                    # print(len(gen_tokens),'>',len(score_dict['green_token_mask']))
                    break
                decoded = tokenizer.decode(i)
                is_green = int(score_dict['green_token_mask'][index])
                if i not in are_tokens_green:
                    if is_green == 1:
                        are_tokens_green[i] = [decoded,1,0]
                    else:
                        are_tokens_green[i] = [decoded,0,1]
                else:
                    if is_green == 1:
                        are_tokens_green[i][1] += 1
                    else:
                        are_tokens_green[i][2] += 1

            # with simple hash's context width of 1, for some reason len(gen_tokens) is always 1 more than # tokens scored???
            # print(len(gen_tokens),score_dict['num_tokens_scored']) 
            gen_length = [score_dict['num_tokens_scored'],score_dict_unique['num_tokens_scored']]
            too_short = False
            if gen_length[0] < args.test_min_tokens:
                print(f"Warning: generation {idx+1} is too short to test. ({gen_length[0]} < {args.test_min_tokens})")
                too_short = True

            # same as detector.detect(gen_tokens), detector.unidetect(gen_tokens); using regular detector and "unique" detector
            z_score = [score_dict['z_score'], score_dict_unique['z_score']]

            p_value = [score_dict['p_value'], score_dict_unique['p_value']]

            unique_threshold = args.threshold # too strict?
            wm_pred = [1 if z_score[0] > args.threshold else 0,
                    1 if z_score[1] > unique_threshold else 0]
            time_completed = int(time())
            print(f'Avg generation time: {(time_completed-initial_time)/(1+idx-num_cur_outputs):.2f}s/it') # since tqdm inaccurate when initial num_cur_outputs > 0

            outputs.append(json.dumps({
                "time_completed": time_completed,
                "prefix": prefix,
                "gold_completion": gold_completion,
                "gen_completion": gen_text,
                "too_short": too_short,
                "z-score": z_score,
                "p-value": p_value,
                "wm_pred": wm_pred,
                "gen_length": gen_length,
                "num_green": num_green,
                "are_tokens_green": are_tokens_green
            }))

            if (idx + 1) % 1 == 0: # originally 100 (dump every 100 outputs), but changed to allow immediate observation
                # write_file(output_file, outputs) # obsolete since I'm not batch writing (with-block already opened file)
                f.write("\n".join(outputs) + "\n") # changed to only open output file in append mode once with a single with-block
                f.flush() # to see outputs immediately (originally implicitly upon each with-block closing upon write_file return)
                outputs = []
                # print('Writing took',int(time())-time_completed,'seconds')

        if outputs:
            # write_file(output_file, outputs) # obsolete since I'm not batch writing (with-block already opened file)
            f.write("\n".join(outputs) + "\n") # changed to only open output file in append mode once with a single with-block
            f.flush() # to see outputs immediately (originally implicitly upon each with-block closing upon write_file return)

    print("Finished!")
    # drive.flush_and_unmount()

# if __name__ == "__main__":
parser = argparse.ArgumentParser()

dataset = "Adaptive" # "OpenGen"
parser.add_argument("--dataset", type=str, default=dataset)
parser.add_argument("--model_name", type=str, default="facebook/opt-1.3b") # baffo32/decapoda-research-llama-7B-hf
parser.add_argument("--strength", type=float, default=2.0) # aka delta, default was 2.0, set to 0 for unwatermarked model
parser.add_argument("--fraction", type=float, default=0.5) # aka gamma, default for Kirchenbauer was 0.25, changed to match Unigram
parser.add_argument("--max_new_tokens", type=int, default=300) # default for Kirchenbauer was 200, changed to match Unigram
# extended implementation uses greedy sampling, so no args for multinomial sampling nor beam search
parser.add_argument("--threshold", type=float, default=4.0) # aka tau, but NOT the same as tau calculated for unique detector, not same as for Unigram
# parser.add_argument("--wm_key", type=int, default=0) # hash key default contained in extended_watermark_processor.py
parser.add_argument("--test_min_tokens", type=int, default=25) # informal minimum from paper

parser.add_argument(
    "--seeding_scheme",
    type=str,
    default="simple_1", # same as in minimal implementation
    help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
)
parser.add_argument(
    "--normalizers",
    type=str,
    default="",
    help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
)
parser.add_argument(
    "--generation_seed",
    type=int,
    default=123,
    help="Seed for setting the torch global rng prior to generation.",
)
parser.add_argument(
    "--use_sampling",
    type=str2bool,
    default=True,
    help="Whether to generate using multinomial sampling.",
)
parser.add_argument(
    "--sampling_temp",
    type=float,
    default=0.7,
    help="Sampling temperature to use when generating using multinomial sampling.",
)
parser.add_argument(
    "--n_beams",
    type=int,
    default=1,
    help="Number of beams to use for beam search. 1 is normal greedy decoding",
)
# no need for bigrams argument because we use both detectors anyway

parser.add_argument("--prompt_file", type=str, default=f"./data/{dataset}/test.jsonl") #
parser.add_argument("--output_dir", type=str, default=f"./data/{dataset}/")
parser.add_argument("--num_test", type=int, default=2000)
parser.add_argument("--avoid_same_file", type=int, default=0) # 0 is false (note: still will not override already written lines)
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1") # https://stackoverflow.com/questions/48796169/how-to-fix-ipykernel-launcher-py-error-unrecognized-arguments-in-jupyter

args = parser.parse_args()
main(args)