import torch
import torch.nn as nn
import argparse
import os
import random
import numpy as np
import time
import json
import hubconf
from quant import *
from tqdm import trange, tqdm
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from diffusers import DDIMPipeline
from data.imagenet import build_imagenet_data, build_imagenet64_data


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.no_grad()
def validate_model(val_loader, model, device=None, print_freq=100, ddim=False):
    if ddim:
        fid = FrechetInceptionDistance(feature=2048)
        fid.reset()
        for i, (images, _) in enumerate(val_loader):
            images = images.to(device)
            fid.update(model(images))
        return fid.compute()
    else:
        if device is None:
            device = next(model.parameters()).device
        else:
            model.to(device)
        model.eval()
        batch_time = AverageMeter('Time', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, top5],
            prefix='Test: ')

        # switch to evaluate mode

        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        return top1.avg


def get_train_samples(train_loader, num_samples):
    train_data = []
    for batch in train_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples]


def generate_images_for_fid(pipe, label_count_path, t=1, save_dir='/home/ofekglick/BRECQ/generated_dataset'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(label_count_path, 'r') as f:
        dict_data = json.load(f)
    for label, label_count in tqdm(dict_data.items()):
        label = int(label)
        label_count = int(label_count)
        batch_size = label_count // 10
        if label == 0:
            continue
        image = pipe(num_inference_steps=t, batch_size=batch_size, class_labels=[label])
        for j in range(len(image.images)):
            image.images[j].save(os.path.join(save_dir, f"imagenet64_label_{label}_{j}.png"))
    print("Done generating images")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='running parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # general parameters for data and model
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parser.add_argument('--arch', default='cd_imagenet64', type=str, help='dataset name',
                        choices=['cd_imagenet64',])
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size for data loader')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loader')
    parser.add_argument('--data_path', default='', type=str, help='path to ImageNet data', required=True)
    parser.add_argument('--t_for_diffuser', default=1, type=int, help='Number of timesteps for diffuser')
    parser.add_argument('--original',action='store_true', help='Use the original model')
    # quantization parameters
    parser.add_argument('--n_bits_w', default=8, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--channel_wise', action='store_true', help='apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=8, type=int, help='bitwidth for activation quantization')
    parser.add_argument('--act_quant', action='store_true', help='apply activation quantization')
    parser.add_argument('--disable_8bit_head_stem', action='store_true')
    parser.add_argument('--test_before_calibration', action='store_true')

    # weight calibration parameters
    parser.add_argument('--num_samples', default=1024, type=int, help='size of the calibration dataset')
    parser.add_argument('--iters_w', default=2000, type=int, help='number of iteration for adaround')
    parser.add_argument('--weight', default=0.01, type=float,
                        help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--sym', action='store_true', help='symmetric reconstruction, not recommended')
    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')
    parser.add_argument('--step', default=20, type=int, help='record snn output per step')

    # activation calibration parameters
    parser.add_argument('--iters_a', default=5000, type=int, help='number of iteration for LSQ')
    parser.add_argument('--lr', default=4e-4, type=float, help='learning rate for LSQ')
    parser.add_argument('--p', default=2.4, type=float, help='L_p norm minimization for LSQ')
    parser.add_argument('--base_path', default='/home/ofekglick/BRECQ', type=str,)
    args = parser.parse_args()
    print(args)
    # seed_all(args.seed)
    base_path = args.base_path
    label_count_path = os.path.join(base_path, 'imagenet_64/train/label_count.json')

    # build imagenet data loader
    train_loader = build_imagenet64_data(
        data_path=os.path.join(base_path, 'imagenet_64'),
    )

    # load model
    cnn, scheduler, pipe = eval('hubconf.{}(pretrained=True)'.format(args.arch))
    cnn_reg, scheduler_reg, pipe_reg = eval('hubconf.{}(pretrained=True)'.format(args.arch))
    cnn.cuda()
    cnn.eval()
    cnn_reg.cuda()
    cnn_reg.eval()
    config_name = f'config_w{args.n_bits_w}_a{args.n_bits_a}_act{args.act_quant}_t{args.t_for_diffuser}'
    # GENERATE IMAGES FOR FID USING REGULAR CONSISTENCY MODEL
    if args.original:
        print("hello")
        generate_images_for_fid(
            pipe_reg,
            label_count_path,
            t=args.t_for_diffuser,
            save_dir=os.path.join(base_path, f'original_t{args.t_for_diffuser}_generated_dataset'),
        )

    # build quantization parameters
    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 'scale_method': 'mse'}
    aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.act_quant}
    print('Weight quantization parameters: {}'.format(wq_params))
    print('Activation quantization parameters: {}'.format(aq_params))
    print(f"Mode: {cnn}")
    if args.arch == 'cd_imagenet64':
        qnn = QuantModel(model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params, ddim=True)
    else:
        qnn = QuantModel(model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.cuda()
    qnn.eval()
    if args.arch == 'cd_imagenet64':
        pipe.unet = qnn
    if not args.disable_8bit_head_stem:
        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()
    pipe.to('cuda:0')
    cali_data = get_train_samples(train_loader, num_samples=args.num_samples)
    device = next(qnn.parameters()).device

    # Initialize weight quantization parameters
    qnn.set_quant_state(True, False)
    pipe(num_inference_steps=1)

    # Kwargs for weight rounding calibration
    kwargs = dict(
        cali_data=cali_data,
        iters=args.iters_w,
        weight=args.weight,
        asym=True,
        b_range=(args.b_start, args.b_end),
        warmup=args.warmup,
        act_quant=False,
        opt_mode='mse',
    )


    def recon_model(model: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(name))
                    block_reconstruction(qnn, module, **kwargs)
            else:
                recon_model(module)


    # Start calibration
    print("Reconstructing the model")
    recon_model(qnn)
    qnn.set_quant_state(weight_quant=True, act_quant=False)
    print("Generating images AFTER calibrating the weight quantization parameters")
    generate_images_for_fid(
        pipe,
        label_count_path,
        save_dir=os.path.join(base_path, 'quantized_model_only_weight_cali_4_bit_generated_dataset'),
    )
    print("DONE GENERATING IMAGES AFTER CALIBRATING THE WEIGHT QUANTIZATION PARAMETERS")

    if args.act_quant:
        # Initialize activation quantization parameters
        qnn.set_quant_state(True, True)
        with torch.no_grad():
            # _ = qnn(cali_data[:64].to(device))
            pipe(num_inference_steps=1)
        # Disable output quantization because network output
        # does not get involved in further computation
        qnn.disable_network_output_quantization()
        # Kwargs for activation rounding calibration
        kwargs = dict(
            cali_data=cali_data,
            iters=args.iters_a,
            act_quant=True,
            opt_mode='mse',
            lr=args.lr,
            p=args.p,
        )
        recon_model(qnn)
        qnn.set_quant_state(weight_quant=True, act_quant=True)
        print("Generating images AFTER calibrating the weight and activation quantization parameters")
        print("W: {} A: {}".format(args.n_bits_w, args.n_bits_a))
        generate_images_for_fid(
            pipe,
            label_count_path,
            save_dir=os.path.join(base_path, f'quantized_model_{config_name}'),
        )
        print("DONE GENERATING IMAGES AFTER CALIBRATING THE WEIGHT AND ACTIVATION QUANTIZATION PARAMETERS")

