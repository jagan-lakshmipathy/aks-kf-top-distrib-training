import os
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import random_split
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn, optim
from collections import OrderedDict
from tqdm import tqdm

torch.manual_seed(0)

PER_REPLICA_BATCH_SIZE = 8
LEARNING_RATE = 0.001


def setup(rank, world_size, backend):
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def load_data(local_rank, data_dir="./data"):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    if local_rank != 0:
        torch.distributed.barrier()

    if local_rank == 0:
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )

        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform
        )
        torch.distributed.barrier()
    else:
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=False, transform=transform
        )

        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=False, transform=transform
        )
    return trainset, testset


def train(model, optimizer, criterion, trainloader, epoch, use_cuda, gpu_id):
    model.train()
    total_loss = 0
    epoch_steps = 0
    for batch_idx, (images, labels) in tqdm(enumerate(trainloader)):
        optimizer.zero_grad()
        if use_cuda:
            images = images.cuda(gpu_id, non_blocking=True)
            labels = labels.cuda(gpu_id, non_blocking=True)

        pred = model(images)

        loss = criterion(pred, labels)
        loss.backward()

        total_loss += loss.item()
        epoch_steps += 1

        optimizer.step()

    return total_loss / epoch_steps


def evaluate(model, criterion, valloader, epoch, use_cuda, gpu_id):
    # Validation loss
    total_loss = 0.0
    epoch_steps = 0
    total = 0
    correct = 0

    model.eval()
    for i, data in enumerate(valloader, 0):
        with torch.no_grad():
            inputs, labels = data
            if use_cuda:
                inputs, labels = inputs.cuda(gpu_id), labels.cuda(gpu_id)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            epoch_steps += 1

    val_acc = correct / total
    val_loss = total_loss / epoch_steps

    return val_loss, val_acc


def test_accuracy(net, testset, use_cuda, device="cpu"):
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if use_cuda:
                images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def run(gpu_id, use_cuda, args):
    """
    This is a single process that is linked to a single GPU

    :param gpu_id: The id of the GPU on the current node
    :param world_size: Total number of processes across nodes
    :param args:
    :return:
    """
    if use_cuda:
        torch.cuda.set_device(gpu_id)


    # The overall rank of this GPU process across multiple nodes
    #global_process_rank = args.nr * args.gpus + gpu_id
    global_process_rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    learning_rate = args.lr
    momentum = args.momentum


    print(f"Running DDP model on Global Process with Rank: {global_process_rank }.")
    setup(global_process_rank, world_size, args.backend)

    model = Net()
    if use_cuda:
        print('Using CUDA with {}'.format(args.backend))
        model.cuda(gpu_id)
        ddp_model = DDP(model, device_ids=[gpu_id])
    else:
        print('Using CPU')
        ddp_model = DDP(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Load training data
    trainset, testset = load_data(gpu_id)
    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_subset, num_replicas=world_size, rank=global_process_rank
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=PER_REPLICA_BATCH_SIZE,
        sampler=train_sampler,
        num_workers=8,
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=PER_REPLICA_BATCH_SIZE, shuffle=True, num_workers=8
    )

    for epoch in range(args.epochs):
        train_loss = train(ddp_model, optimizer, criterion, trainloader, epoch, use_cuda, gpu_id)
        print("train_loss", train_loss, ' epoch: ', epoch, ' rank: ', global_process_rank, ' local_rank: ', gpu_id)

        val_loss, val_acc = evaluate(ddp_model, criterion, valloader, epoch, use_cuda, gpu_id)
        print("val_loss", val_loss, ' epoch: ', epoch, ' rank: ', global_process_rank, ' local_rank: ', gpu_id)
        print("val_acc", val_acc, ' epoch: ', epoch, ' rank: ', global_process_rank, ' local_rank: ', gpu_id)


    test_acc = test_accuracy(model, testset, use_cuda, f"cuda:{gpu_id}")
    print("test_acc", test_acc, ' epoch: ', epoch, ' rank: ', global_process_rank, ' local_rank: ', gpu_id)

    cleanup()


class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int)
    #parser.add_argument("-b", "--backend", type=str, default="nccl")
    parser.add_argument(
        "-n",
        "--nodes",
        default=1,
        type=int,
        metavar="N",
        help="total number of compute nodes",
    )
    parser.add_argument(
        "-g", "--gpus", default=1, type=int, help="number of gpus per node"
    )
    parser.add_argument(
        "-nr", "--nr", default=0, type=int, help="ranking within the nodes, starts at 0"
    )
    parser.add_argument(
        "--epochs",
        default=2,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training",
    )
    parser.add_argument(
        "--backend",
        type=str,
        help="Distributed backend",
        choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
        default=dist.Backend.GLOO,
    )
    return parser.parse_args()


def main():
    args = get_args()
    print('args =>', args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cpu'

    if use_cuda:
        print("Using CUDA")
        torch.manual_seed(0)
        device = 'cuda'
        if args.backend != dist.Backend.NCCL:
            print(
                "Warning. Please use `nccl` distributed backend for the best performance using GPUs"
            )

    # Attach model to the device.
    model = Net().to(device)

    print("Using distributed PyTorch with {} backend".format(args.backend))
    # Set distributed training environment variables to run this training script locally.
    if "WORLD_SIZE" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "1234"

    print(f"World Size: {os.environ['WORLD_SIZE']}. Rank: {os.environ['RANK']}")

    print('LOCAL_RANK', dist.get_node_local_rank(0))
    print('WORLD_SIZE', int(os.environ['WORLD_SIZE']))
    print('WORLD_RANK', int(os.environ['RANK']))
    print('MASTER_ADDR', os.environ['MASTER_ADDR'])
    print('MASTER_PORT', int(os.environ['MASTER_PORT']))
    print('LOCAL_RANK', dist.get_node_local_rank(0))

    run(dist.get_node_local_rank(0), use_cuda, args)



if __name__ == "__main__":
    main()
