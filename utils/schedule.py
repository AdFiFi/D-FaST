from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math


def get_cosine_annealing_schedule_with_warmup(optimizer: Optimizer, eta_max: float, eta_min: float, num_warmup_steps: int,
                                              num_training_steps: int, last_epoch: int = -1, num_cycles: float = 0.5):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(eta_min, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress) +
                                   eta_min / eta_max * progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def init_schedule(optimizer, args, t_total):
    if args.schedule == 'cos':
        schedule = CosineAnnealingLR(optimizer, eta_min=args.target_learning_rate, T_max=t_total)
    elif args.schedule == 'cos_w':
        schedule = get_cosine_annealing_schedule_with_warmup(optimizer, eta_max=args.learning_rate,
                                                             eta_min=args.target_learning_rate,
                                                             num_warmup_steps=args.warmup_steps,
                                                             num_training_steps=t_total)
    elif args.schedule == 'linear':
        schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                   num_training_steps=t_total)
    elif args.schedule == 'one_cycle':
        schedule = OneCycleLR(optimizer,
                              max_lr=args.max_learning_rate,
                              epochs=args.num_epochs,
                              steps_per_epoch=t_total // args.num_epochs,
                              pct_start=0.2,
                              div_factor=args.max_learning_rate/args.learning_rate,
                              final_div_factor=1000)
    else:
        schedule = None
    return schedule
