import os
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from sklearn.metrics import roc_curve, auc

from setup import config, seed_worker
from utils.util_model import EmbedNet, TripletNet
import utils.util_path as PATH
from utils.util_vis import draw_roc_curve
from utils.util_metric import AverageMeter
from datasets import get_dataset
from backbones import get_backbone
from models import get_model


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device_id = config.gpu_ids[rank]
    torch.cuda.set_device(device_id)

    train_dataset = get_dataset(config.data, config=config, data_path=config.train_data_path)
    test_dataset = get_dataset(config.data, config=config, data_path=config.test_data_path)

    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, config.batch_size, num_workers=config.num_workers, sampler=train_sampler, pin_memory=True, worker_init_fn=seed_worker)
    test_loader = torch.utils.data.DataLoader(test_dataset, config.batch_size, num_workers=config.num_workers, sampler=test_sampler, pin_memory=True, worker_init_fn=seed_worker)

    backbone = get_backbone(config.backbone)
    model = get_model(config.model)
    embed_net = EmbedNet(backbone, model)
    triplet_net = DDP(TripletNet(embed_net).to(device_id), device_ids=[device_id])

    criterion = torch.nn.TripletMarginWithDistanceLoss(margin=config.margin, distance_function=F.pairwise_distance)
    optimizer = torch.optim.Adam(triplet_net.parameters(), lr=config.learning_rate)

    os.makedirs(PATH.CHECKPOINT, exist_ok=True)
    os.makedirs(PATH.VISUALIZATION, exist_ok=True)

    def train():
        triplet_net.train()
        losses = AverageMeter()

        for i, (anc, pos, neg) in enumerate(train_loader):
            anc, pos, neg = anc.to(device_id), pos.to(device_id), neg.to(device_id)
            optimizer.zero_grad()
            anc_feat, pos_feat, neg_feat = triplet_net(anc, pos, neg)
            loss = criterion(anc_feat, pos_feat, neg_feat)
            loss.backward()
            optimizer.step()

            losses.update(loss, anc.size(0))
        return losses.avg

    def validate():
        triplet_net.eval()
        losses = AverageMeter()
        dist_poses = AverageMeter()
        dist_neges = AverageMeter()
        y_true = []
        y_scores = []

        with torch.no_grad():
            for i, (anc, pos, neg) in enumerate(test_loader):
                anc, pos, neg = anc.to(device_id), pos.to(device_id), neg.to(device_id)
                anc_feat, pos_feat, neg_feat = triplet_net(anc, pos, neg)
                loss = criterion(anc_feat, pos_feat, neg_feat)
                dist_pos = F.pairwise_distance(anc_feat, pos_feat).cpu().numpy()
                dist_neg = F.pairwise_distance(anc_feat, neg_feat).cpu().numpy()

                losses.update(loss.item(), anc.size(0))
                dist_poses.update(np.mean(dist_pos), anc.size(0))
                dist_neges.update(np.mean(dist_neg), anc.size(0))
                y_true.extend([1] * anc.size(0))
                y_true.extend([0] * anc.size(0))
                y_scores.extend(dist_pos)
                y_scores.extend(dist_neg)

        fpr, tpr, thresholds = roc_curve(y_true, -np.array(y_scores))  # -y_scores because smaller distances should correspond to larger scores
        roc_auc = auc(fpr, tpr)
        return losses.avg, dist_poses.avg, dist_neges.avg, roc_auc, fpr, tpr

    for epoch in range(1, config.total_epoch + 1):
        train_sampler.set_epoch(epoch)
        train_loss = train()
        avg_loss, avg_dist_pos, avg_dist_neg, roc_auc, fpr, tpr = validate()
        if rank == 0:
            print(f'[Epoch {epoch}] Train loss {train_loss:.4f}')
            print(f'[Epoch {epoch}] Validation loss {avg_loss:.4f}')
            print(f'[Epoch {epoch}] Average distance with positive sample: {avg_dist_pos:.4f}')
            print(f'[Epoch {epoch}] Average distance with negative sample: {avg_dist_neg:.4f}')
            print(f'[Epoch {epoch}] ROC AUC: {roc_auc:.4f}')
            draw_roc_curve(fpr, tpr, os.path.join(PATH.VISUALIZATION, f'roc_curve_e{epoch}.png'), roc_auc)
            torch.save(triplet_net.state_dict(), os.path.join(PATH.CHECKPOINT, f'{config.backbone}_{config.model}_checkpoint_e{epoch}.pth'))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()