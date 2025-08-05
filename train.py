import os

#os.environ["CUDA_VISIBLE_DEVICES"]="7"

import argparse
from train_framework import train_models
import pickle
import json

def main():

    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument("--train_path", type=str, help="path to img jpg files", default= r'/data/zhuyu/data/google_test/tile/image/')
    parser.add_argument("--save_path", type=str, help="Model save path.", default= r'./model/HCFNet')
    parser.add_argument("--batch_size", type=int, default=8, help="train batch size")
    parser.add_argument("--num_epochs", type=int, default=150, help="number of epochs")
    parser.add_argument(
        "--use_pretrained", type=bool, default=False, help=" whether to use pretrained checkpoint."
    )
    parser.add_argument(
        "--pretrained_model_path", type=str,
        default='./model/google_all/61.pt', help="If use_pretrained checkpoint, provide checkpoint pathway."
    )
    parser.add_argument(
        "--augment_dataset", type=bool,
        default=False, 
        help="Whether to augment the training dataset(color space augmentation is performed regardless of setting)."
    )
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float,
                        metavar='W', help='default weight decay')
    parser.add_argument('--stepsize', default=20, type=int,
                        metavar='SS', help='learning rate step size')
    parser.add_argument('--gamma', '--gm', default=0.1, type=float,
                        help='learning rate decay parameter: Gamma')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--itersize', default=2, type=int,
                        metavar='IS', help='iter size')
                        
    args = parser.parse_args()
    
    

    history = train_models(train_path=args.train_path, save_path=args.save_path, batch_size=args.batch_size, num_epochs=args.num_epochs, use_pretrained=args.use_pretrained,
                 pretrained_model_path=args.pretrained_model_path, augment_dataset=args.augment_dataset, momentum=args.momentum, weight_decay=args.weight_decay,stepsize=args.stepsize, gamma=args.gamma, lr=args.lr,itersize=args.itersize)
    
    # Save history file
    model_name = 'HCFNet'
    try:
        with open(f'{model_name}_train_loss.json', 'w') as f:
            json.dump(history, f)
        #with open(f'{model_name}_train_loss.pkl', 'wb') as f:
            #pickle.dump(history, f)
        print(f"成功将 '{model_name}' 的训练历史保存到 {model_name}_train_loss.json")
    except Exception as e:
        print(f"保存 history 文件失败: {e}")

if __name__ == "__main__":
    main()








