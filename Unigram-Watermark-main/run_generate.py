import argparse
from tqdm import tqdm
import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LogitsProcessorList, BitsAndBytesConfig
from gptwm import GPTWatermarkLogitsWarper


def read_file(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f.read().strip().split("\n")]


def write_file(filename, data):
    with open(filename, "a") as f:
        f.write("\n".join(data) + "\n")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use gpu
    # print(f"Model will run on: {device}")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, llm_int8_enable_fp32_cpu_offload=True, bnb_4bit_compute_dtype=torch.float16)
    # device_map = 'auto'
    device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 0, 'model.layers.17': 0, 'model.layers.18': 0, 'model.layers.19': 0, 'model.layers.20': 0, 'model.layers.21': 0, 'model.layers.22': 0, 'model.layers.23': 0, 'model.layers.24': 'cpu', 'model.layers.25': 'cpu', 'model.layers.26': 'cpu', 'model.layers.27': 'cpu', 'model.layers.28': 'cpu', 'model.layers.29': 'cpu', 'model.layers.30': 'cpu', 'model.layers.31': 'cpu', 'model.norm': 'cpu', 'model.rotary_emb': 'cpu', 'lm_head': 'cpu'}

    output_file = f"{args.output_dir}/{args.model_name.replace('/', '-')}_strength_{args.strength}_frac_{args.fraction}_len_{args.max_new_tokens}_num_{args.num_test}.jsonl"
    if 'llama' in args.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name, torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, torch_dtype=torch.float16)

    # adding offload folder for weights, "auto" should primarily use gpu, 4bit quantization
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map=device_map, offload_folder='./offload/', quantization_config=quantization_config)
    print(model.hf_device_map)
    # device_map="auto" (unquantized): {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 'cpu', 'model.layers.3': 'cpu', 'model.layers.4': 'cpu', 'model.layers.5': 'cpu', 'model.layers.6': 'cpu', 'model.layers.7': 'cpu', 'model.layers.8': 'cpu', 'model.layers.9': 'cpu', 'model.layers.10': 'cpu', 'model.layers.11': 'disk', 'model.layers.12': 'disk', 'model.layers.13': 'disk', 'model.layers.14': 'disk', 'model.layers.15': 'disk', 'model.layers.16': 'disk', 'model.layers.17': 'disk', 'model.layers.18': 'disk', 'model.layers.19': 'disk', 'model.layers.20': 'disk', 'model.layers.21': 'disk', 'model.layers.22': 'disk', 'model.layers.23': 'disk', 'model.layers.24': 'disk', 'model.layers.25': 'disk', 'model.layers.26': 'disk', 'model.layers.27': 'disk', 'model.layers.28': 'disk', 'model.layers.29': 'disk', 'model.layers.30': 'disk', 'model.layers.31': 'disk', 'model.norm': 'disk', 'model.rotary_emb': 'disk', 'lm_head': 'disk'}
    # device_map="auto" (quantized): {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 0, 'model.layers.17': 0, 'model.layers.18': 0, 'model.layers.19': 0, 'model.layers.20': 0, 'model.layers.21': 0, 'model.layers.22': 0, 'model.layers.23': 0, 'model.layers.24': 'cpu', 'model.layers.25': 'cpu', 'model.layers.26': 'cpu', 'model.layers.27': 'cpu', 'model.layers.28': 'cpu', 'model.layers.29': 'cpu', 'model.layers.30': 'cpu', 'model.layers.31': 'cpu', 'model.norm': 'cpu', 'model.rotary_emb': 'cpu', 'lm_head': 'cpu'}
    model.eval()

    # print(tokenizer.vocab_size)
    # print(model.vocab_size) # should be same?
    # print(tokenizer.get_vocab())

    watermark_processor = LogitsProcessorList([GPTWatermarkLogitsWarper(fraction=args.fraction,
                                                                        strength=args.strength,
                                                                        vocab_size=model.config.vocab_size,
                                                                        watermark_key=args.wm_key)])

    data = read_file(args.prompt_file)
    num_cur_outputs = len(read_file(output_file)) if os.path.exists(output_file) else 0

    outputs = []

    for idx, cur_data in tqdm(enumerate(data), total=min(len(data), args.num_test)):
        if idx < num_cur_outputs or len(outputs) >= args.num_test:
            continue

        if "gold_completion" not in cur_data and 'targets' not in cur_data:
            continue
        elif "gold_completion" in cur_data:
            prefix = cur_data['prefix']
            gold_completion = cur_data['gold_completion']
        else:
            prefix = cur_data['prefix']
            gold_completion = cur_data['targets'][0]

        batch = tokenizer(prefix, truncation=True, return_tensors="pt").to(device) # inputs should be on same device as model
        num_tokens = len(batch['input_ids'][0])

        with torch.inference_mode():
            generate_args = {
                **batch,
                'logits_processor': watermark_processor,
                'output_scores': True,
                'return_dict_in_generate': True,
                'max_new_tokens': args.max_new_tokens,
            }

            if args.beam_size is not None:
                generate_args['num_beams'] = args.beam_size
            else:
                generate_args['do_sample'] = True
                generate_args['top_k'] = args.top_k
                generate_args['top_p'] = args.top_p

            generation = model.generate(**generate_args)
            gen_text = tokenizer.batch_decode(generation['sequences'][:, num_tokens:], skip_special_tokens=True)

        outputs.append(json.dumps({
            "prefix": prefix,
            "gold_completion": gold_completion,
            "gen_completion": gen_text
        }))

        if (idx + 1) % 1 == 0: #100
            write_file(output_file, outputs)
            outputs = []

    write_file(output_file, outputs)
    print("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("--model_name", type=str, default="facebook/opt-125m")
    parser.add_argument("--model_name", type=str, default="baffo32/decapoda-research-llama-7B-hf") # decapoda-research/llama-7b-hf no longer exists
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--wm_key", type=int, default=0)
    parser.add_argument("--prompt_file", type=str, default="./data/OpenGen/inputs.jsonl")
    parser.add_argument("--output_dir", type=str, default="./data/OpenGen/")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--num_test", type=int, default=500)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)

    args = parser.parse_args()
    main(args)
