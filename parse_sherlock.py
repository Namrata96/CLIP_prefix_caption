import torch
import os
os.environ["LD_LIBRARY_PATH"] = "/ext3/miniconda3/envs/clip_prefix_caption/lib:" + os.environ["LD_LIBRARY_PATH"]
print("========" + os.environ["LD_LIBRARY_PATH"])
import skimage.io as io
import clip
from PIL import Image
from PIL import ImageDraw
import pickle
import json
import os
from tqdm import tqdm
import argparse

def url2filepath(args, url):
        if 'VG_' in url:
            return args.vg_dir + '/'.join(url.split('/')[-2:])
        else:
            # http://s3-us-west-2.amazonaws.com/ai2-rowanz/vcr1images/lsmdc_3023_DISTRICT_9/3023_DISTRICT_9_01.21.02.808-01.21.16.722@5.jpg
            if 'vcr1images' in args.vcr_dir:
                return args.vcr_dir + '/'.join(url.split('/')[-2:])
            else:
                return args.vcr_dir + '/'.join(url.split('/')[-3:])

def main(args):
    clip_model_type = args.clip_model_type
    with open('/scratch/nm3571/multimodal/data/sherlock/new_sherlock_train.json', 'r') as f:
        train_data = json.load(f)
    with open('/scratch/nm3571/multimodal/data/sherlock/new_sherlock_val.json', 'r') as f:
        val_data = json.load(f)


    print("-------BEFORE PROCESSING--------")
    print("Length of training data", len(train_data))
    print("Length of validation data", len(val_data))
    # print("Length of test data", len(test_data))
    print("Total", len(train_data)+len(val_data))

    url_dict = dict()
    for i, item in enumerate(val_data):
        url = item["inputs"]["image"]["url"]
        if url not in url_dict:
            url_dict[url] = []
        url_dict[url].append(item)

    for url in url_dict:
        train_data.extend(url_dict[url])

    print("-------AFTER PROCESSING--------")
    print("Length of train data", len(train_data))
    print("Length of val data", 0)
    # print("Length of test data", len(test_data))
    print("Total", len(train_data))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"/scratch/nm3571/multimodal/models/clipcap/sherlock/oscar_split_{clip_model_name}_train.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    # with open('/scratch/nm3571/multimodal/data/clipcap/sherlock/train_caption.json', 'r') as f:
    #     data = json.load(f)
    data = train_data
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    j=0
    for i in tqdm(range(len(data))):
        d = data[i]
        # img_id = d["image_id"]
        filename = url2filepath(args, d['inputs']['image']['url'])
        # f"/scratch/nm3571/multimodal/data/clipcap/sherlock/COCO_val2014_{int(img_id):012d}.jpg"
        # f"./data/coco/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        if not os.path.isfile(filename):
            raise Exception(f"NOT A VALID FILE NAME {filename}")
            # filename = f"./data/coco/val2014/COCO_val2014_{int(img_id):012d}.jpg"
        image = Image.open(filename)

        image = image.convert('RGBA')
         #highlight mode
        overlay = Image.new('RGBA', image.size, '#00000000')
        draw = ImageDraw.Draw(overlay, 'RGBA')
       
        for bbox in d['inputs']['bboxes']:
            x = bbox['left']
            y = bbox['top']
           # highlight mode
            draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])],
                            fill='#ff05cd3c', outline='#05ff37ff', width=3)
           
        highlighted_image = Image.alpha_composite(image, overlay)

        # instance_id = d['instance_id']
        # highlighted_image.save(f"/scratch/nm3571/multimodal/output/clipcap/sherlock/images_with_bbox/{filename[:-4]}_{instance_id}.jpg")
        
        image = preprocess(highlighted_image).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        d["clip_embedding"] = j
        j = j + 1
        all_embeddings.append(prefix)
        all_captions.append(d)
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument(
        '--vcr_dir',
        default='images/',
        help='directory with all of the VCR image data, contains, e.g., movieclips_Lethal_Weapon')

    parser.add_argument(
        '--vg_dir',
        default='images/',
        help='directory with visual genome data, contains VG_100K and VG_100K_2')
    
    args = parser.parse_args()
    if args.vcr_dir[-1] != '/':
        args.vcr_dir += '/'
    if args.vg_dir[-1] != '/':
        args.vg_dir += '/'
    exit(main(args))

