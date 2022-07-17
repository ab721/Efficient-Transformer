import os
import tqdm
import torch
import wandb
import random
import numpy as np
import pandas as pd
from pathlib import Path
from albumentations import *
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score, accuracy_score

import data_pipeline, model, dice_loss

SEED = 42
random.seed(SEED)                          
np.random.seed(SEED)                       
torch.manual_seed(SEED)                    
torch.cuda.manual_seed(SEED)               
torch.cuda.manual_seed_all(SEED)           
torch.backends.cudnn.deterministic = True

def list_diff(list1, list2):
    list_dif = [i for i in list1 + list2 if i not in list1 or i not in list2]
    return list_dif

def train(config):
    
    wandb.init(config=config, project="Efficient Transformer")

    mdl = model.Efficient_Transformer(swin_weight_path = config['swin_weight_path'], decoder_channels = config['decoder_channels'], 
                                      decoder_scale_factors = config['decoder_scale_factors'], swin_drop_rate = config['swin_drop_rate'], 
                                      swin_attn_drop_rate = config['swin_attn_drop_rate'], swin_drop_path_rate = config['swin_drop_path_rate'], 
                                      decoder_widths = config['decoder_widths'], num_classes = config['num_classes'])

    loss_function = dice_loss.DiceLoss()

    validation_text = f"{config['data_directory']}/{config['subset'][:-1].capitalize()}_compiled.txt"
    dtype_dic= {'ImageId': str,'fold_id' : int}
    df_cvfolds = pd.read_csv(validation_text, usecols=['ImageId','fold_id'], dtype = dtype_dic)
    total_files = [i[:-4] for i in sorted(os.listdir(f"{config['data_directory']}/Images"))]
    given_fold = int(config['subset'][-1])
    test_set = df_cvfolds[df_cvfolds.fold_id == given_fold].ImageId.tolist()
    train_set = list_diff(total_files, test_set)

    output_directory = f"{config['output_directory']}/Checkpoints/Efficient_Transformer/{config['model_identifier']}"
    Path(output_directory).mkdir(parents = True, exist_ok = True)

    best_iou = 0.0

    mdl = mdl.cuda()

    train_dataloader = data_pipeline.get_dataloader(data_directory = config['data_directory'], augmentation = config['train_augmentation'], image_names = train_set, shuffle = True,
                                                    batch_size = config['batch_size'], num_workers = config['num_workers'])


    optimizer = torch.optim.AdamW(mdl.parameters(), lr = config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = config['lr_decay_epochs'], gamma = config['lr_decay_rate'])

    for e in range(config['num_epochs']):
        
        progress_bar = tqdm.tqdm(total = len(train_dataloader) * train_dataloader.batch_size)
        progress_bar.set_description(f'epoch {e+1}')
  
        for l, (img, msk) in enumerate(train_dataloader):

            mdl.train()
            img = img.cuda()
            msk = msk.cuda()
            optimizer.zero_grad()
            out = mdl(img)
            loss = loss_function(out, msk)
            loss.backward()
            optimizer.step()
            progress_bar.update(img.size(0))
            
        scheduler.step()

        progress_bar.close()

        jaccard0, jaccard1, precision0, precision1, recall0, recall1, f10, f11, accuracy = test(config = config, mdl = mdl, epoch = e + 1, test_set = test_set)
        metrics = {'jaccard_background': jaccard0, 'jaccard_corrosion': jaccard1, 
                   'precision_background': precision0, 'precision_corrosion': precision1,
                   'recall_background': recall0, 'recall_corrosion': recall1,
                   'f1_background': f10, 'f1_corrosion': f11,
                   'accuracy_overall': accuracy}
        wandb.log(metrics)

        if jaccard1 > best_iou:
            best_iou = jaccard1
            torch.save({'model': mdl.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 
                         'epoch': e+1, 'metrics': metrics}, f"{output_directory}/best_mdl.pth")



def test(config, mdl, epoch, test_set):
    
    test_dataloader = data_pipeline.get_dataloader(data_directory = config['data_directory'], augmentation = config['test_augmentation'], image_names = test_set, shuffle = False,
                                                    batch_size = 1, num_workers = config['num_workers'])

    true_values = []
    predicted_values = []


    progress_bar = tqdm.tqdm(total = len(test_dataloader) * test_dataloader.batch_size)
    progress_bar.set_description(f'epoch {epoch}')

    for (img, msk) in test_dataloader:

        mdl.eval()
        img = img.cuda()
        msk_ = msk.numpy().astype(np.uint8)
        out = (mdl(img) > config['threshold']).detach().cpu().numpy()
        true_values.append(msk_.ravel())
        predicted_values.append(out.ravel())

        progress_bar.update(img.size(0))   

    progress_bar.close()

    true_values = np.array(true_values).ravel()
    predicted_values = np.array(predicted_values).ravel()

    jaccard0, jaccard1 = jaccard_score(true_values, predicted_values, average = None)
    precision0, precision1 = precision_score(true_values, predicted_values, average = None)
    recall0, recall1 = recall_score(true_values, predicted_values, average = None)
    f10, f11 = f1_score(true_values, predicted_values, average = None)
    accuracy = accuracy_score(true_values, predicted_values)
    
    print(f'*************************** For {epoch} *******************************')
    print(f'The Jaccard score is {jaccard0}, {jaccard1}')
    print(f'The F1 score is {f10}, {f11}')
    print(f'The Precision is {precision0}, {precision1}')
    print(f'The Recall is {recall0}, {recall1}')
    print(f'The Accuracy is {accuracy}')

    return jaccard0, jaccard1, precision0, precision1, recall0, recall1, f10, f11, accuracy


if __name__ == '__main__':

    config = dict(
                train_augmentation = Compose([HorizontalFlip(), RandomRotate90(), ChannelShuffle(), CoarseDropout(), ColorJitter(), 
                                            FancyPCA(), GaussianBlur(), Resize(384, 384,interpolation=cv2.INTER_NEAREST), 
                                            Normalize()], p=1.0),
                test_augmentation = Compose([Resize(384, 384,interpolation=cv2.INTER_NEAREST), Normalize()], p=1.0),
                data_directory = '../input/corrosion-segmentation/Resized/Resized',
                subset = 'kfold1',
                batch_size = 12,
                model_identifier = 'first_run',
                output_directory = ".", #for kaggle, otherwise, same as data_directory
                swin_weight_path = '../input/swin-weights/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth',
                decoder_channels = 256,
                decoder_scale_factors = [8, 4, 2, 1],
                swin_drop_rate = 0,
                swin_attn_drop_rate = 0,
                swin_drop_path_rate = 0.2,
                decoder_widths = [184, 336, 704, 1352],
                num_classes = 1,
                learning_rate = 2e-5,
                num_epochs = 20,
                lr_decay_epochs = [50, 100, 150, 200],
                lr_decay_rate = 0.1,
                threshold = 0.5,
                num_workers = 4
                )

    train.train(config)