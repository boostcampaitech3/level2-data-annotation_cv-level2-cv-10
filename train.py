import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset, SceneTextDataset_CV
from model import EAST

import mlflow


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/Modified'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)

    #### mlflow ####
    parser.add_argument('--experiment_name', type=str, default="")
    ################

    parser.add_argument('--validation', type=int, default=0) # default = no validation

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, experiment_name, validation):
    #### mlflow ####
    if experiment_name:
        mlflow.set_registry_uri("https://0.0.0.0:30001")
        mlflow.set_experiment(experiment_name)
        print(f"Current experiment name: {experiment_name}")
    ################

    # ====== train set =======
    # split='train'으로 넘겨주면 기존 전체 train.json을 사용하게 됩니다. 이 경우에는 val set은 사용하지 않습니다.
    # fold 1 set을 사용하고 싶다면 split='train_1'로 넘겨주시면 됩니다.
    dataset = SceneTextDataset_CV(data_dir, split='train_0', image_size=image_size, crop_size=input_size)
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # ====== val set ======
    if validation:
        # fold 1 set을 사용하고 싶다면 split='val_1'로 넘겨주시면 됩니다.
        dataset_val = SceneTextDataset_CV(data_dir, split='val_0', image_size=image_size, crop_size=input_size, validation=True)
        dataset_val = EASTDataset(dataset_val)
        num_batches_val = math.ceil(len(dataset_val) / batch_size)
        val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    # for early stopping
    patience = 10
    counter = 0

    best_val_loss = 999

    # console color
    color_train = 93 # yellow
    color_val = 96 # skyblue

    for epoch in range(max_epoch):
        # ====== train ======
        model.train()
        # for checking losses per epoch
        epoch_loss, epoch_start = 0, time.time()
        epoch_cls_loss, epoch_angle_loss, epoch_iou_loss = 0., 0., 0. 

        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                # for checking losses per epoch
                epoch_cls_loss += extra_info['cls_loss']
                epoch_angle_loss += extra_info['angle_loss']
                epoch_iou_loss += extra_info['iou_loss']

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss'],
                    'Learning Rate': optimizer.param_groups[0]['lr'],
                }
                pbar.set_postfix(val_dict)
        
        epoch_loss /= num_batches 
        epoch_cls_loss /= num_batches
        epoch_angle_loss /= num_batches
        epoch_iou_loss /= num_batches
        
        print(f'\033[{color_train}m Train {epoch+1}/{max_epoch} - '
                  f'Mean loss: {epoch_loss:.4f}, '
                  f'Cls loss: {epoch_cls_loss:.4f}, '
                  f'Angle loss: {epoch_angle_loss:.4f}, '
                  f'IoU loss: {epoch_iou_loss:.4f} | '
                  f'Elapsed time: {timedelta(seconds=time.time() - epoch_start)}' + '\033[0m')

        #### mlflow ####
        if experiment_name:
            mlflow.log_metric("loss/train", epoch_loss)
            mlflow.log_metric("cls_loss/train", epoch_cls_loss)
            mlflow.log_metric("angle_loss/train", epoch_angle_loss)
            mlflow.log_metric("iou_loss/train", epoch_iou_loss)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'])
        ################

        # ====== val ======
        if validation:
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                # for checking losses per epoch
                epoch_loss, epoch_start = 0, time.time()
                epoch_cls_loss, epoch_angle_loss, epoch_iou_loss = 0., 0., 0. 

                with tqdm(total=num_batches_val) as pbar:
                    for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                        pbar.set_description('[Epoch {}]'.format(epoch + 1))

                        loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)

                        loss_val = loss.item()
                        epoch_loss += loss_val

                        # for checking losses per epoch
                        epoch_cls_loss += extra_info['cls_loss']
                        epoch_angle_loss += extra_info['angle_loss']
                        epoch_iou_loss += extra_info['iou_loss']

                        pbar.update(1)
                        val_dict = {
                            'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                            'IoU loss': extra_info['iou_loss']
                        }
                        pbar.set_postfix(val_dict)

                epoch_loss /= num_batches_val 
                epoch_cls_loss /= num_batches_val
                epoch_angle_loss /= num_batches_val
                epoch_iou_loss /= num_batches_val

                print(f'\033[{color_val}m Validation {epoch+1}/{max_epoch} - '
                  f'Mean loss: {epoch_loss:.4f}, '
                  f'Best val loss: {best_val_loss:.4f} | '
                  f'Elapsed time: {timedelta(seconds=time.time() - epoch_start)}' + '\033[0m')

                #### mlflow ####
                if experiment_name:
                    mlflow.log_metric("loss/val", epoch_loss)
                    mlflow.log_metric("cls_loss/val", epoch_cls_loss)
                    mlflow.log_metric("angle_loss/val", epoch_angle_loss)
                    mlflow.log_metric("iou_loss/val", epoch_iou_loss)
                ################

                # validation accuracy가 향상될수록 모델을 저장합니다.
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    if not osp.exists(model_dir):
                        os.makedirs(model_dir)
                    ckpt_fpath = osp.join(model_dir, f'best_val_loss.pth')
                    torch.save(model.state_dict(), ckpt_fpath)
                    print(f'\033[{color_val}m' + f"Best val loss at epoch {epoch+1}! Saving the model to {ckpt_fpath}..." + '\033[0m')
                    counter = 0
                else:
                    counter += 1
                # patience 횟수 동안 성능 향상이 없을 경우 Early stopping
                if counter > patience:
                    print("Early stopping...")
                    break

        scheduler.step()

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)


def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
