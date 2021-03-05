from os import makedirs

from data import DATASETS
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import Dataset
import argparse

from archs import load_architecture
from utilities import get_flow_directory, get_loss_and_acc, compute_losses, \
    save_files, save_files_final, AtParams, compute_gradient, get_hessian_eigenvalues, DEFAULT_PHYS_BS
from data import load_dataset, take_first


def rk_step(network: nn.Module, loss_fn: nn.Module, dataset: Dataset, step_size: float,
            physical_batch_size: int = DEFAULT_PHYS_BS):
    """ Take a Runge-Kutta step with a given step size. """
    theta = parameters_to_vector(network.parameters())

    def f(x: torch.Tensor):
        with AtParams(network, x):
            fx = -compute_gradient(network, loss_fn, dataset, physical_batch_size=physical_batch_size)
        return fx

    k1 = f(theta)
    k2 = f(theta + (step_size / 2) * k1)
    k3 = f(theta + (step_size / 2) * k2)
    k4 = f(theta + step_size * k3)

    theta_next = theta + (step_size / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    vector_to_parameters(theta_next, network.parameters())


def rk_advance_time(network: nn.Module, loss_fn: nn.Module, dataset: Dataset, T: float,
                    rk_step_size: float, physical_batch_size: int):
    """ Using the Runge-Kutta algorithm, numerically integrate the gradient flow ODE for time T, using a given
     Runge-Kutta step size."""
    T_remaining = T
    while T_remaining > 0:
        this_step_size = min(rk_step_size, T_remaining)
        rk_step(network, loss_fn, dataset, this_step_size, physical_batch_size)
        T_remaining -= rk_step_size


def main(dataset: str, arch_id: str, loss: str, max_time: float, tick: float, neigs=0, physical_batch_size=1000,
         abridged_size: int = 5000, eig_freq: int = -1, iterate_freq: int = -1, save_freq: int = -1, alpha: float = 1.0,
         nproj: int = 0, loss_goal: float = None, acc_goal: float = None, max_step_size: int = 999, seed: int = 0):
    directory = get_flow_directory(dataset, arch_id, seed, loss, tick)
    print(f"output directory: {directory}")
    makedirs(directory, exist_ok=True)

    train_dataset, test_dataset = load_dataset(dataset, loss)
    abridged_train = take_first(train_dataset, abridged_size)

    loss_fn, acc_fn = get_loss_and_acc(loss)

    max_steps = int(max_time / tick)

    torch.manual_seed(seed)
    network = load_architecture(arch_id, dataset).cuda()

    torch.manual_seed(7)
    projectors = torch.randn(nproj, len(parameters_to_vector(network.parameters())))

    times, train_loss, test_loss, train_acc, test_acc = \
        torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), \
        torch.zeros(max_steps), torch.zeros(max_steps)
    iterates = torch.zeros(max_steps // iterate_freq if iterate_freq > 0 else 0, len(projectors))
    eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)

    sharpness = float('inf')

    for step in range(0, max_steps):
        times[step] = step * tick
        train_loss[step], train_acc[step] = compute_losses(network, [loss_fn, acc_fn], train_dataset,
                                                           physical_batch_size)
        test_loss[step], test_acc[step] = compute_losses(network, [loss_fn, acc_fn], test_dataset, physical_batch_size)

        if eig_freq != -1 and step % eig_freq == 0:
            eigs[step // eig_freq, :] = get_hessian_eigenvalues(network, loss_fn, abridged_train, neigs=neigs,
                                                                physical_batch_size=physical_batch_size)
            sharpness = eigs[step // eig_freq, 0]
            print(f"sharpness = {sharpness}", flush=True)

        if iterate_freq != -1 and step % iterate_freq == 0:
            iterates[step // iterate_freq, :] = projectors.mv(parameters_to_vector(network.parameters()).cpu().detach())

        if save_freq != -1 and step % save_freq == 0:
            save_files(directory, [("eigs", eigs[:step // eig_freq]), ("iterates", iterates[:step // iterate_freq]),
                                   ("train_loss", train_loss[:step]), ("test_loss", test_loss[:step]),
                                   ("train_acc", train_acc[:step]), ("test_acc", test_acc[:step]),
                                   ("times", times[:step])])

        if (loss_goal is not None and train_loss[step] < loss_goal) or \
                (acc_goal is not None and train_acc[step] > acc_goal):
            break

        print(f"{times[step]:.3f}\t{train_loss[step]:.3f}\t{train_acc[step]:.3f}"
              f"\t{test_loss[step]:.3f}\t{test_acc[step]:.3f}", flush=True)

        if (loss_goal != None and train_loss[step] < loss_goal) or (acc_goal != None and train_acc[step] > acc_goal):
            break

        rk_step_size = min(alpha / sharpness, max_step_size)
        rk_advance_time(network, loss_fn, train_dataset, tick, rk_step_size, physical_batch_size)

    save_files_final(directory, [("time", times[:step + 1]), ("eigs", eigs[:(step + 1) // eig_freq]),
                                 ("iterates", iterates[:(step + 1) // iterate_freq]),
                                 ("train_loss", train_loss[:step + 1]),
                                 ("test_loss", test_loss[:step + 1]), ("train_acc", train_acc[:step + 1]),
                                 ("test_acc", test_acc[:step + 1])])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using gradient flow.")
    parser.add_argument("dataset", type=str, choices=DATASETS, help="which dataset to train")
    parser.add_argument("arch_id", type=str, help="which network architectures to train")
    parser.add_argument("loss", type=str, choices=["ce", "mse"], help="which loss function to use")
    parser.add_argument("tick", type=float,
                        help="the train / test losses and accuracies will be computed and saved every tick units of time")
    parser.add_argument("max_time", type=float, help="the maximum time (ODE time, not wall clock time) to train for")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help=" the Runge-Kutta step size is min(alpha / [estimated sharpness], max_step_size).")
    parser.add_argument("--max_step_size", type=float, default=999,
                        help=" the Runge-Kutta step size is min(alpha / [estimated sharpness], max_step_size)")
    parser.add_argument("--seed", type=int, help="the random seed used when initializing the network weights",
                        default=0)
    parser.add_argument("--beta", type=float, help="momentum parameter (used if opt = polyak or nesterov)")
    parser.add_argument("--physical_batch_size", type=int,
                        help="the maximum number of examples that we try to fit on the GPU at once", default=1000)
    parser.add_argument("--acc_goal", type=float,
                        help="terminate training if the train accuracy ever crosses this value")
    parser.add_argument("--loss_goal", type=float, help="terminate training if the train loss ever crosses this value")
    parser.add_argument("--neigs", type=int, help="the number of top eigenvalues to compute")
    parser.add_argument("--eig_freq", type=int, default=-1,
                        help="the frequency at which we compute the top Hessian eigenvalues (-1 means never)")
    parser.add_argument("--nproj", type=int, default=0, help="the dimension of random projections")
    parser.add_argument("--iterate_freq", type=int, default=-1,
                        help="the frequency at which we save random projections of the iterates")
    parser.add_argument("--abridged_size", type=int, default=5000,
                        help="when computing top Hessian eigenvalues, use an abridged dataset of this size")
    parser.add_argument("--save_freq", type=int, default=-1,
                        help="the frequency at which we save resuls")
    parser.add_argument("--save_model", type=bool, default=False,
                        help="if 'true', save model weights at end of training")
    args = parser.parse_args()

    main(dataset=args.dataset, arch_id=args.arch_id, loss=args.loss, max_time=args.max_time, tick=args.tick,
         neigs=args.neigs, physical_batch_size=args.physical_batch_size, abridged_size=args.abridged_size,
         eig_freq=args.eig_freq, iterate_freq=args.iterate_freq, save_freq=args.save_freq, nproj=args.nproj,
         loss_goal=args.loss_goal, acc_goal=args.acc_goal, seed=args.seed)
