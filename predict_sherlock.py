
from train import ClipCaptionModel, ClipCaptionPrefix
import json
import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
)

import skimage.io as io
import PIL.Image
import argparse

def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                    entry_length=67, temperature=1., stop_token: str = '.'):

        model.eval()
        stop_token_index = tokenizer.encode(stop_token)[0]
        tokens = None
        scores = None
        device = next(model.parameters()).device
        seq_lengths = torch.ones(beam_size, device=device)
        is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
        with torch.no_grad():
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)
                    generated = model.gpt.transformer.wte(tokens)
            for i in range(entry_length):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                logits = logits.softmax(-1).log()
                if scores is None:
                    scores, next_tokens = logits.topk(beam_size, -1)
                    generated = generated.expand(beam_size, *generated.shape[1:])
                    next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(beam_size, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    logits[is_stopped] = -float(np.inf)
                    logits[is_stopped, 0] = 0
                    scores_sum = scores[:, None] + logits
                    seq_lengths[~is_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                    next_tokens_source = next_tokens // scores_sum.shape[1]
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)
                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                    generated = generated[next_tokens_source]
                    scores = scores_sum_average * seq_lengths
                    is_stopped = is_stopped[next_tokens_source]
                next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                if is_stopped.all():
                    break
        scores = scores / seq_lengths
        output_list = tokens.cpu().numpy()
        output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
        order = scores.argsort(descending=True)
        output_texts = [output_texts[i] for i in order]
        return output_texts


def generate2(
            model,
            tokenizer,
            tokens=None,
            prompt=None,
            embed=None,
            entry_count=1,
            entry_length=67,  # maximum number of words
            top_p=0.8,
            temperature=1.,
            stop_token: str = '.',
    ):
        model.eval()
        generated_num = 0
        generated_list = []
        stop_token_index = tokenizer.encode(stop_token)[0]
        filter_value = -float("Inf")
        device = next(model.parameters()).device

        with torch.no_grad():

            for entry_idx in trange(entry_count):
                if embed is not None:
                    generated = embed
                else:
                    if tokens is None:
                        tokens = torch.tensor(tokenizer.encode(prompt))
                        tokens = tokens.unsqueeze(0).to(device)

                    generated = model.gpt.transformer.wte(tokens)

                for i in range(entry_length):

                    outputs = model.gpt(inputs_embeds=generated)
                    logits = outputs.logits
                    logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                        ..., :-1
                                                        ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = filter_value
                    next_token = torch.argmax(logits, -1).unsqueeze(0)
                    next_token_embed = model.gpt.transformer.wte(next_token)
                    if tokens is None:
                        tokens = next_token
                    else:
                        tokens = torch.cat((tokens, next_token), dim=1)
                    generated = torch.cat((generated, next_token_embed), dim=1)
                    if stop_token_index == next_token.item():
                        break

                output_list = list(tokens.squeeze().cpu().numpy())
                output_text = tokenizer.decode(output_list)
                generated_list.append(output_text)

        return generated_list[0]

def url2filepath(args, url):
        if 'VG_' in url:
            return args.vg_dir + '/'.join(url.split('/')[-2:])
        else:
            # http://s3-us-west-2.amazonaws.com/ai2-rowanz/vcr1images/lsmdc_3023_DISTRICT_9/3023_DISTRICT_9_01.21.02.808-01.21.16.722@5.jpg
            if 'vcr1images' in args.vcr_dir:
                return args.vcr_dir + '/'.join(url.split('/')[-2:])
            else:
                return args.vcr_dir + '/'.join(url.split('/')[-3:])

def evaluate(args):
    #Caption prediction
    predictions = list()
    model_path = args.model_path

    #@title CLIP model + GPT2 tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(args.clip_model_type, device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


    #Load model weights

    prefix_length = 10

    model = ClipCaptionModel(prefix_length)

    model.load_state_dict(torch.load(model_path, map_location='cpu')) 

    model = model.eval() 
    model = model.to(device)

    with open(args.test_file, 'r') as f:
        test_data = json.load(f)

    image_paths = []
    for item in test_data:
        image_paths.append(url2filepath(args, item["inputs"]["image"]["url"]))

    for path, item in zip(image_paths, test_data):
        image = io.imread(path)
        pil_image = PIL.Image.fromarray(image)
        #pil_img = Image(filename=UPLOADED_FILE)
        # display(pil_image)

        image = preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            # if type(model) is ClipCaptionE2E:
            #     prefix_embed = model.forward_image(image)
            # else:
            prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
            prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
        if args.do_beam_search:
            generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
        else:
            generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)

        predictions.append({'prediction': generated_text_prefix, 'data':item})
        print('\n')
        print(generated_text_prefix)
    
    s = json.dump(predictions, open(args.save_file,"w"))
    # with open(args.predictions_file,"w") as f:
    #     f.write(s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--model_path', default="./coco.pt", help="Path to model weights")
    parser.add_argument('--save_file', default="/scratch/nm3571/", help="File to save generated predictions")
    parser.add_argument('--do_beam_search', action='store_false')
    parser.add_argument('--pretrain_model', default="coco", choices=('coco', 'conceptual captions'))
    parser.add_argument(
        '--vcr_dir',
        default='images/',
        help='directory with all of the VCR TEST image data, contains, e.g., movieclips_Lethal_Weapon')

    parser.add_argument(
        '--vg_dir',
        default='images/',
        help='directory with visual genome TEST data, contains VG_100K and VG_100K_2')
    parser.add_argument('--test_file', default="/scratch/nm3571/multimodal/data/sherlock/new_sherlock_test.json", help="File containing examples to run inference on")
    
    args = parser.parse_args()
    if args.vcr_dir[-1] != '/':
        args.vcr_dir += '/'
    if args.vg_dir[-1] != '/':
        args.vg_dir += '/'
    exit(evaluate(args))
