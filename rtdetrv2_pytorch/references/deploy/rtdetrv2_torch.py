"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time

import torch
import torch.nn as nn 
import torchvision.transforms as T

import numpy as np 
from PIL import Image, ImageDraw

from src.core import YAMLConfig
from src.zoo.rtdetr.hybrid_encoder import TransformerEncoderLayer
from fvcore.nn import FlopCountAnalysis


def draw(images, labels, boxes, scores, thrh = 0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j,b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(),2)}", fill='blue', )

        im.save(f'results_{i}.jpg')


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    # ----------------------------------- #
    # For training MPO
    # ----------------------------------- #
    # cfg.model.encoder.encoder[0].layers[0].from_pretrained(cfg.model.encoder.encoder[0].layers[0].moe)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes=torch.tensor([640,640]).to("cpu")):
            flops = FlopCountAnalysis(self.model, images)
            print("FLOPS: ", flops.total())
            flops_data = dict(flops.by_module())
            print(flops_data['encoder.encoder'])
            import pdb; pdb.set_trace()
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None].to(args.device)

    # ----------------------------------- #
    # For pruning
    # ----------------------------------- #
    # import torch_pruning as tp
    # imp = tp.importance.GroupNormImportance(p=2)
    # example_inputs = torch.randn(1,3,640,640).to("cpu")
    # num_heads = {}
    # for m in model.modules():
    #     if isinstance(m, nn.MultiheadAttention):
    #         num_heads[m] = 8
    # pruner = tp.pruner.MetaPruner(
    #     cfg.model,
    #     example_inputs,
    #     importance=imp,
    #     # num_heads=num_heads,
    #     # customized_pruners={nn.MultiheadAttention: tp.pruner.function.MultiheadAttentionPruner},
    #     global_pruning=True,
    #     pruning_ratio=0.0, # default pruning ratio
    #     pruning_ratio_dict = {(cfg.model.encoder.encoder[0].layers[0].linear1): 0.4}, 
    # )
    # pruner.step()
    import torch.nn.utils.prune as prune
    prune.l1_unstructured(model.model.encoder.encoder[0].layers[0].linear1, name="weight",  amount=0.75)
    prune.l1_unstructured(model.model.encoder.encoder[0].layers[0].linear2, name="weight",  amount=0.75)
    print("Sparsity:", 100 * float(torch.sum(model.model.encoder.encoder[0].layers[0].linear1.weight == 0)) / model.model.encoder.encoder[0].layers[0].linear1.weight.nelement(), "%")
    model.model.encoder.encoder[0].layers[0].sparse_weight = model.model.encoder.encoder[0].layers[0].linear1.weight.to_sparse()
    model.model.encoder.encoder[0].layers[0].bias = model.model.encoder.encoder[0].layers[0].linear1.bias

    start_time = time.time()
    output = model(im_data, orig_size)
    print(f'Params: {sum(p.numel() for p in model.parameters())}')
    print(f'Inference time: {time.time()-start_time:.3f}s')
    labels, boxes, scores = output

    draw([im_pil], labels, boxes, scores)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-f', '--im-file', type=str, )
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
