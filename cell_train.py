"""
Train a diffusion model on embeded gene expression data.
"""

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.cell_datasets_muris import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (     
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

from guided_diffusion.train_util import TrainLoop

import torch
import numpy as np
import random

def main():
    setup_seed(1234) 
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir='checkpoint/logs/'+args.model_name)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        ae_dir=args.ae_dir,
        num_gene=args.num_genes,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        model_name=args.model_name,
        save_dir = 'checkpoint'
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir='/data1/lep/Workspace/guided-diffusion/data/tabula_muris/all.h5ad',
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0001,
        lr_anneal_steps=1000000,
        batch_size=128,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=200000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        model_name="my_model",
        class_cond=False,
        ae_dir='checkpoint/AE/muris_all/model_seed=0_step=800000.pt',
        num_genes=18996,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    main()
