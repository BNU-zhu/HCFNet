import os

#os.environ["CUDA_VISIBLE_DEVICES"]="5,6,7"

from time import time
from logger import Logger
import glob
import random
from dataloader_2 import ImageFolder_2
from torch.utils.data import DataLoader
from tqdm import tqdm
from accuracy import evaluate
from accelerate import Accelerator
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
from accelerate.state import DistributedType
from losses_new import *
from network.HCFNet import HCFNet
from torch import nn
from torch.autograd import Variable as V
print(f"Available GPUs: {torch.cuda.device_count()}")

train_loss_history = []
# Define loss function
def define_loss(deep_supervision_weights=[0.3, 0.3, 0.4, 0.5, 0.6, 1.0]):
    return LossForHCFNet_Advanced(deep_supervision_weights)#LossBsiNet(weights)

# Train step function
def loss_train(model, inputs, targets, criterion, optimizer, accelerator):
    optimizer.zero_grad()
    with torch.set_grad_enabled(True):
        outputs = model(inputs)
        loss = criterion(outputs[0], outputs[1], targets[0], targets[1])
        accelerator.backward(loss)
        optimizer.step()
    return loss

# Main training function
def train_models(train_path, save_path, batch_size, num_epochs, use_pretrained, pretrained_model_path,
                 augment_dataset, momentum, weight_decay, stepsize, gamma, lr, itersize):

    # Initialize logger and tensorboard writer
    mylog = Logger('logs/' + 'model.log')
    #log_path = os.path.join(save_path, "summary")
    #writer = SummaryWriter(log_dir=log_path)
    tic = time()

    # Initialize Accelerator
    accelerator = Accelerator(mixed_precision="no")
    device = accelerator.device
    print(f"Using device: {accelerator.device}") 
    if accelerator.distributed_type == DistributedType.MULTI_GPU:
        print("Using MULTI_GPU setup")
    else:
        print(f"Distributed Type: {accelerator.distributed_type}")

    # Prepare dataset
    train_file_names = glob.glob(os.path.join(train_path, "*.tif"))
    random.shuffle(train_file_names)
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_file_names]
    train_file, val_file = train_test_split(img_ids, test_size=0.01, random_state=41)

    if augment_dataset:
        train_dataset = Augment(ImageFolder_2(train_path, train_file))
    else:
        train_dataset = ImageFolder_2(train_path, train_file)
    
    val_dataset = ImageFolder_2(train_path, val_file)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=32, drop_last=True, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False, pin_memory=True)

    # Initialize model, optimizer, and scheduler
    model = HCFNet(num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(1e10), eta_min=1e-5)
    criterion = define_loss()

    # Prepare model, optimizer, and data loaders for Accelerate
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # Load pretrained model if applicable
    if use_pretrained:
        print("Loading Pretrained Model: {}".format(os.path.basename(pretrained_model_path)))
        state_dict = torch.load(pretrained_model_path)
        new_state_dict = {"module." + k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)

        '''
        # Remove unnecessary prefix 'module.' from state_dict
        if 'module.' in list(checkpoint.keys())[0]:
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        '''

    train_loss_history = []
    for epoch in tqdm(range(1, num_epochs + 1)):
        global_step = epoch * len(train_loader)
        running_loss = 0.0
        counter = 0
        for i, (img_file_name, inputs, sample1, sample2) in enumerate(
            tqdm(train_loader)
        ):
            model.train()
            with torch.no_grad():
                inputs = V(inputs.cuda()).float()
                sample1 = V(sample1.cuda())#.float()
                sample2 = V(sample2.cuda())#.float()         
            counter += 1
            targets = [sample1, sample2]
            loss = loss_train(model, inputs, targets, criterion, optimizer, accelerator)
            #writer.add_scalar("loss", loss.item(), epoch)

            running_loss += loss.item() * inputs.size(0)
        scheduler.step()

        epoch_loss = running_loss / len(train_file_names)
        train_loss_history.append(epoch_loss)
        #print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.6f}")
        accelerator.print(f"Epoch {epoch+1} | Train Loss: {epoch_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.7f}")
        
        
        if epoch % 1 == 0:
            #reg_val_acc, bou_val_acc = evaluate(model, val_loader)#device
            mylog.write('********\n')
            mylog.write('epoch:' + str(epoch) + '    time:' + str(int(time() - tic)) + '\n')
            mylog.write('train_loss:' + str(epoch_loss) + '\n')
            #mylog.write('reg_val_acc:' + str(reg_val_acc) + '\n')
            #mylog.write('bou_val_acc:' + str(bou_val_acc) + '\n')
            
        if epoch % 10 == 0:
            
            model_to_save = accelerator.unwrap_model(model)
            torch.save(model_to_save.state_dict(), os.path.join(save_path, str(epoch) + ".pt"))
            
    #with open(f'{model_name}_train_loss.pkl', 'wb') as f:
    #pickle.dump(train_loss_history, f)
    #mylog.write('Finish!')
    #mylog.close()
    return train_loss_history
    


