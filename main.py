import os

import wandb

from config import init_config
from trainers import *
from utils import *
from torch import distributed


# os.environ['CUDA_VISIBLE_DEVICES'] = "1,3"
logger = logging.getLogger(__name__)
# os.environ['WANDB_API_KEY'] = "75f175974297dec15f4f8ea7970a66002d3c2cc6"
# os.environ['WANDB_API_KEY'] = "cd4441a5fcdd740b84b45deb6890ecb376bddecb"
# os.environ['WANDB_API_KEY'] = "local-1a6c67774093a56b310d8313b6821f2d98e59678"
# os.environ['WANDB_API_KEY'] = "local-961e0d861bcf53caabe769f4ce59fe7a1ff8eede"
# os.environ['WANDB_MODE'] = "offline"


def cross_subject(args):
    if args.do_train:
        results = Recorder()
        local_rank = 0
        if args.do_parallel:
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            rank = int(os.environ['RANK'])
            distributed.init_process_group('nccl', world_size=world_size, rank=rank)
            # distributed.init_process_group('gloo', world_size=self.world_size, rank=self.rank)
            torch.cuda.set_device(local_rank)
        for i in range(args.num_repeat):
        # for i in range(1, 2):
            group_name = f"{args.model}" \
                         f"_{args.dataset}" \
                         f"_{args.batch_size}" \
                         f"{f'sparsity-{args.sparsity}' if 'FaSTP' in args.model else ''}" \
                         f'F{args.frequency}D{args.D}F{args.num_kernels}P{args.p1}={args.p2}_dp{args.dropout}' \
                         f"_w{args.window_size}" \
                         f"{'_mp' if args.mix_up else ''}" \
                         f"-cross"

            run = wandb.init(project=args.project, entity=args.wandb_entity, reinit=True,
                             group=f"{group_name}", tags=[args.dataset])

            trainer = eval(args.model + 'Trainer')(args, local_rank=local_rank, task_id=i)
            init_logger(f'{args.log_dir}/train_{args.model}_{args.dataset}.log')
            logger.info(f"{'#'*10} Repeat:{i} {'#'*10}")
            trainer.train()
            results.add_record(trainer.best_result)

            run.finish()
        results.save(os.path.join(args.model_dir, args.model, 'results.json'))
    elif args.do_test:
        trainer = eval(args.model + 'Trainer')(args)
        init_logger(f'{args.log_dir}/test_{args.model}_{args.dataset}.log')
        trainer.load_model()
        trainer.evaluate()


def within_subject(args):
    if args.do_train:
        local_rank = 0
        group_name = ''
        if args.do_parallel:
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            rank = int(os.environ['RANK'])
            distributed.init_process_group('nccl', world_size=world_size, rank=rank)
            # distributed.init_process_group('gloo', world_size=self.world_size, rank=self.rank)
            torch.cuda.set_device(local_rank)
        for subject_id in range(1, args.subject_num+1):
            best_results = Recorder()
            final_results = Recorder()
            for i in range(args.num_repeat):
                group_name = f"{args.model}" \
                             f"_{args.dataset}" \
                             f"_{args.batch_size}" \
                             f"{f'sparsity-{args.sparsity}' if 'FaSTP' in args.model else ''}" \
                             f'F{args.frequency}D{args.D}F{args.num_kernels}P{args.p1}={args.p2}_dp{args.dropout}' \
                             f"_w{args.window_size}" \
                             f"{'_mp' if args.mix_up else ''}" \
                             f"-within"

                run = wandb.init(project=args.project, entity=args.wandb_entity, reinit=True,
                                 group=f"{group_name}", tags=[args.dataset, f'id_{subject_id}'])

                trainer = eval(args.model + 'Trainer')(args, local_rank=local_rank, task_id=i, subject_id=subject_id)
                init_logger(f'{args.log_dir}/train_{args.model}_{args.dataset}.log')
                logger.info(f"{'#'*10} Subject:{i} {'#'*10}")
                trainer.train()
                best_results.add_record(trainer.best_result)
                final_results.add_record(trainer.test_result)
                run.finish()
            best_results.save(os.path.join(args.model_dir, args.model, 'best_results.json'))
            final_results.save(os.path.join(args.model_dir, args.model, 'final_results.json'))
            run = wandb.init(project=args.project, entity=args.wandb_entity, reinit=True,
                             group=f"{group_name}-results", tags=[args.dataset])
            wandb.log({f"best {k}": v for k, v in best_results.get_avg().items()})
            wandb.log(final_results.get_avg())
            run.finish()

    elif args.do_test:
        trainer = eval(args.model + 'Trainer')(args)
        init_logger(f'{args.log_dir}/test_{args.model}_{args.dataset}.log')
        trainer.load_model()
        trainer.evaluate()


def parameters(args):
    trainer = eval(args.model + 'Trainer')(args)
    total = sum([param.nelement() for param in trainer.model.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))


if __name__ == '__main__':
    Args = init_config()
    # if Args.within_subject:
    #     within_subject(Args)
    # else:
    #     cross_subject(Args)
    parameters(Args)
