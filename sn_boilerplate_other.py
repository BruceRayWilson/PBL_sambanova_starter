import argparse
import sys
from typing import Tuple


import torch
import torch.nn as nn
import torchvision

from sambaflow import samba

import sambaflow.samba.utils as utils
from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.utils.pef_utils import get_pefmeta
from sambaflow.samba.utils.dataset.mnist import dataset_transform


def prepare_dataloader(args: argparse.Namespace) -> Tuple[torch.utils.data.DataLoader]:
    """Train the model on RDU using the MNIST dataset (images and labels)."""
    train_dataset = torchvision.datasets.MNIST(root=f'{args.data_folder}',
                                               train=True,
                                               transform=dataset_transform(args),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root=f'{args.data_folder}',
                                              train=False,
                                              transform=dataset_transform(args))

    # Data loader (input pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader


def train(args: argparse.Namespace, model: nn.Module, optimizer: samba.optim.SGD) -> None:
    """Train the model."""
    train_loader, test_loader = prepare_dataloader(args)

    total_step = len(train_loader)
    hyperparam_dict = {"lr": args.lr, "momentum": args.momentum, "weight_decay": args.weight_decay}
    for epoch in range(args.num_epochs):
        avg_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            sn_images = samba.from_torch(images, name='image', batch_dim=0)
            sn_labels = samba.from_torch(labels, name='label', batch_dim=0)

            loss, outputs = samba.session.run(input_tensors=[sn_images, sn_labels],
                                              output_tensors=model.output_tensors,
                                              hyperparam_dict=hyperparam_dict)
            loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)
            avg_loss += loss.mean()

            if (i + 1) % 10000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.num_epochs, i + 1, total_step,
                                                                         avg_loss / (i + 1)))

        samba.session.to_cpu(model)
        test_acc = 0.0
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0
            for images, labels in test_loader:
                loss, outputs = model(images, labels)
                loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)
                total_loss += loss.mean()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

            test_acc = 100.0 * correct / total
            print('Test Accuracy: {:.2f}'.format(test_acc),
                  ' Loss: {:.4f}'.format(total_loss.item() / (len(test_loader))))

        if args.acc_test:
            assert args.num_epochs == 1, "Accuracy test only supported for 1 epoch"
            assert test_acc > 91.0 and test_acc < 92.0, "Test accuracy not within specified bounds."


def test(args: argparse.Namespace, model: nn.Module, inputs: Tuple[samba.SambaTensor],
         outputs: Tuple[samba.SambaTensor]) -> None:
    """Test the model by compairing the Samba and Torch outputs."""
    samba.session.tracing = False
    outputs_gold = model(*inputs)

    outputs_samba = samba.session.run(input_tensors=inputs,
                                      output_tensors=outputs,
                                      data_parallel=args.data_parallel,
                                      reduce_on_rdu=args.reduce_on_rdu)

    # check that all samba and torch outputs match numerically
    for i, (output_samba, output_gold) in enumerate(zip(outputs_samba, outputs_gold)):
        print('samba:', output_samba)
        print('gold:', output_gold)
        samba.utils.assert_close(output_samba, output_gold, f'forward output #{i}', threshold=3e-3)

    if not args.inference:
        # training mode, check two of the gradients
        torch_loss, torch_gemm_out = outputs_gold
        torch_loss.mean().backward()

        # we choose two gradients from different places to test numerically
        gemm1_grad_gold = model.ffn.gemm1.weight.grad
        gemm1_grad_samba = model.ffn.gemm1.weight.sn_grad

        samba.utils.assert_close(gemm1_grad_gold, gemm1_grad_samba, 'ffn__gemm1__weight__grad', threshold=3e-3)
