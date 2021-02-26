from os import makedirs

import torch
import torch.nn as nn
from fire import Fire
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import Dataset

from archs import load_architecture
from utilities import get_modified_flow_directory, get_loss_and_acc, compute_losses, compute_hvp, \
    save_files, save_files_final, compute_gradient, get_hessian_eigenvalues, DEFAULT_PHYS_BS, AtParams
from data import load_dataset, take_first


def compute_modified_gradient(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                              lr: float, physical_batch_size: int):
    gradient = compute_gradient(network, loss_fn, dataset, physical_batch_size)
    return gradient + 0.5 * lr * compute_hvp(network, loss_fn, dataset, gradient, physical_batch_size)


def rk_step(network: nn.Module, loss_fn: nn.Module, dataset: Dataset, gd_lr: float, step_size: float,
            physical_batch_size: int = DEFAULT_PHYS_BS):
    """ Take a Runge-Kutta step with a given step size. """
    theta = parameters_to_vector(network.parameters())

    def f(x: torch.Tensor):
        with AtParams(network, x):
            return -compute_modified_gradient(network, loss_fn, dataset, gd_lr, physical_batch_size=physical_batch_size)

    k1 = f(theta)
    k2 = f(theta + (step_size / 2) * k1)
    k3 = f(theta + (step_size / 2) * k2)
    k4 = f(theta + step_size * k3)

    theta_next = theta + (step_size / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    vector_to_parameters(theta_next, network.parameters())


def rk_advance_time(network: nn.Module, loss_fn: nn.Module, dataset: Dataset, gd_lr: float, T: float,
                    rk_step_size: float, physical_batch_size: int):
    """ Using the Runge-Kutta algorithm, numerically integrate the gradient flow ODE for time T, using a given
     Runge-Kutta step size."""
    T_remaining = T
    while T_remaining > 0:
        this_step_size = min(rk_step_size, T_remaining)
        rk_step(network, loss_fn, dataset, gd_lr, this_step_size, physical_batch_size)
        T_remaining -= rk_step_size


def main(dataset, arch_id, loss, gd_lr, max_time, tick, neigs=0, physical_batch_size=1000, abridged_size=5000,
         eig_freq=-1, iterate_freq=-1, save_freq=-1, alpha=1.0, nproj=0, loss_goal=None,
         acc_goal=None, max_step_size=999, seed=0):
    directory = get_modified_flow_directory(dataset, arch_id, seed, loss, gd_lr, tick)
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
        rk_advance_time(network, loss_fn, train_dataset, gd_lr, tick, rk_step_size, physical_batch_size)

    save_files_final(directory, [("time", times[:step + 1]), ("eigs", eigs[:(step + 1) // eig_freq]),
                                 ("iterates", iterates[:(step + 1) // iterate_freq]),
                                 ("train_loss", train_loss[:step + 1]),
                                 ("test_loss", test_loss[:step + 1]), ("train_acc", train_acc[:step + 1]),
                                 ("test_acc", test_acc[:step + 1])])


if __name__ == "__main__":
    Fire(main)
