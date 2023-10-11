import torch.nn as nn
import warnings,os,random,  torch
import numpy as np
from loss import FocalLoss
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from model import Act_net
from load_dataset import load_dataset
from cal_metrics import  cal_metrics, print_metrics, best_acc_thr
from load_dataset import CelllineDataset
import argparse
import ast


# Define the default values
seed, cell_line, emb_type, max_len, epoch_num, batch_size, patience, threshold, save_path,lr,weight_decay,\
cnn_kernel_size,lstm_hidden_dim,lstm_num_layers,lstm_dropout,l_dropout,\
m, h, d_ff,act_num_of_hid,n_list,repeat_num,act_dropout=\
2023,"K562","onehot",2000,1000,64,100,0.5,os.getcwd(),0.001,0.0,\
3,8,1,0.0,0.0,\
100,6,6,6,[3],12,0.0



# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Command-line parameters')
parser.add_argument('--seed', type=int, default=seed, help='An integer')
parser.add_argument('--cell_line', type=str, default=cell_line, help='A string')
parser.add_argument('--emb_type', type=str, default=emb_type, help='A string')
parser.add_argument('--max_len', type=int, default=max_len, help='An integer')
parser.add_argument('--epoch_num', type=int, default=epoch_num, help='An integer')
parser.add_argument('--batch_size', type=int, default=batch_size, help='An integer')
parser.add_argument('--patience', type=int, default=patience, help='An integer')
parser.add_argument('--threshold', type=float, default=threshold, help='A float')
parser.add_argument('--save_path', type=str, default=save_path, help='A string')
parser.add_argument('--lr', type=float, default=lr, help='A float')
parser.add_argument('--weight_decay', type=float, default=weight_decay, help='A float')

parser.add_argument('--cnn_kernel_size', type=int, default=cnn_kernel_size, help='An integer')
parser.add_argument('--lstm_hidden_dim', type=int, default=lstm_hidden_dim, help='An integer')
parser.add_argument('--lstm_num_layers', type=int, default=lstm_num_layers, help='An integer')
parser.add_argument('--lstm_dropout', type=float, default=lstm_dropout, help='A float')
parser.add_argument('--l_dropout', type=float, default=l_dropout, help='A float')
parser.add_argument('--m', type=int, default=m, help='An integer')
parser.add_argument('--h', type=int, default=h, help='An integer')
parser.add_argument('--d_ff', type=int, default=d_ff, help='An integer')
parser.add_argument('--act_num_of_hid', type=int, default=act_num_of_hid, help='An integer')
parser.add_argument('--n_list', type=str, default=str(n_list), help='A list of integers')
parser.add_argument('--repeat_num', type=int, default=repeat_num, help='An integer')
parser.add_argument('--act_dropout', type=float, default=act_dropout, help='A float')


# Parse the command-line arguments
args = parser.parse_args()

# Convert param4 from string to list of integers using ast.literal_eval
try:
    n_list = ast.literal_eval(args.n_list)
    if not isinstance(n_list, list) or not all(isinstance(item, int) for item in n_list):
        raise ValueError("Invalid input for param4, it must be a list of integers.")
except (SyntaxError, ValueError):
    print("Invalid input for param4, it must be a list of integers.")
    n_list = n_list

seed, cell_line, emb_type, max_len, epoch_num, batch_size, patience, threshold, save_path,lr,weight_decay,\
cnn_kernel_size,lstm_hidden_dim,lstm_num_layers,lstm_dropout,l_dropout,\
m, h, d_ff,act_num_of_hid,repeat_num,act_dropout=\
args.seed, args.cell_line, args.emb_type, args.max_len, args.epoch_num, args.batch_size, args.patience, args.threshold, args.save_path,args.lr,args.weight_decay,\
args.cnn_kernel_size,args.lstm_hidden_dim,args.lstm_num_layers,args.lstm_dropout,args.l_dropout,\
args.m, args.h, args.d_ff,args.act_num_of_hid,args.repeat_num,args.act_dropout




## random seed #
random_seed = seed
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHUSHSEED'] = str(random_seed)



warnings.filterwarnings('ignore')
if torch.cuda.is_available():
    num_workers = 0
    device = torch.device("cuda:" + str(0))
    torch.cuda.set_device(0)
else:
    num_workers = 0
    device = torch.device("cpu")


def initial_model(model):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal(m.weight.data)
            m.bias.data.zero_()
        if isinstance(m, (nn.LSTM)):
            nn.init.orthogonal_(m.weight_ih_l0)
            nn.init.orthogonal_(m.weight_hh_l0)
            nn.init.orthogonal_(m.weight_ih_l0)
            nn.init.orthogonal_(m.weight_hh_l0)
            nn.init.zeros_(m.bias_ih_l0)
            nn.init.zeros_(m.bias_hh_l0)
        elif isinstance(m, (nn.MultiheadAttention)):
            nn.init.xavier_normal_(m.out_proj.weight.data)
            m.out_proj.bias.data.zero_()
    

def train_cv():
    print(f'===================================New Training===================================')
    print("Device: ", device)
    print("Seed: ", seed)
    # Load datasets
    train_dataset, test_dataset = load_dataset(emb_type, max_len, seed)
    # test_dataset = CelllineDataset(['ATTTCTTGGGGGGGGGGGCCC'],np.array(1),emb_type=emb_type,max_len=max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=np.random.seed(seed))
    pos_weight = float(train_dataset.num_non / train_dataset.num_ess)
    # Loss function
    loss = FocalLoss(gamma=2, pos_weight=pos_weight, logits=False, reduction='sum')
    # loss = nn.CrossEntropyLoss()
    # loss = nn.BCEWithLogitsLoss()
    # Train and validation using 5-fold cross validation
    val_auprs, test_auprs = [], []
    val_aucs, test_aucs = [], []
    test_trues, kfold_test_scores = [], []
    kfold_val_trues, kfold_val_scores = [], []
    kfold = 5
    skf = StratifiedKFold(n_splits=kfold, random_state=seed, shuffle=True)
    for i, (train_index, val_index) in enumerate(skf.split(train_dataset.features, train_dataset.labels)):
        print(f'\nStart training CV fold {i+1}:')

        train_sampler, val_sampler = SubsetRandomSampler(train_index), SubsetRandomSampler(val_index)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=num_workers, worker_init_fn=np.random.seed(seed))
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False, num_workers=num_workers, worker_init_fn=np.random.seed(seed))
        # Model
        model=Act_net(d=train_dataset.emb_dim,l=max_len,m=m, h=h,cnn_kernel_size=cnn_kernel_size,lstm_num_layers=lstm_num_layers,lstm_dropout=lstm_dropout,n_list=n_list,
                      act_dropout=act_dropout,lstm_hidden_dim=lstm_hidden_dim, l_dropout=l_dropout,repeat_num=repeat_num,num_of_hid=act_num_of_hid,d_ff=d_ff)
        model = model.to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

        # Adjust learning rate
        def lr_lambda(epoch):
        # adjust learning rate function










            return 1

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=- 1, verbose=True)
        count = 0
        best_val_acc, best_test_acc = .0, .0
        best_val_auc, best_test_auc = .0, .0
        best_val_aupr, best_test_aupr = .0, .0
        best_test_scores = []
        best_model = model


        for epoch in range(epoch_num):
            print(f'\nEpoch [{epoch+1}/{epoch_num}]')

            # Calculate prediction results and losses
            train_trues, train_scores, train_loss = cal_by_epoch(mode='train', model=model, loader=train_loader, loss=loss, optimizer=optimizer)
            val_trues, val_scores, val_loss = cal_by_epoch(mode='val', model=model, loader=val_loader, loss=loss)
            test_trues, test_scores, test_loss = cal_by_epoch(mode='test', model=model, loader=test_loader, loss=loss)
            scheduler.step()

            # Calculate evaluation meteics
            train_metrics = cal_metrics(train_trues, train_scores, threshold)[:]
            val_metrics = cal_metrics(val_trues, val_scores, threshold)[:]
            test_metrics = cal_metrics(test_trues, test_scores, threshold)[:]

            val_auc, val_aupr  = val_metrics[-2], val_metrics[-1]
            test_auc, test_aupr  = test_metrics[-2], test_metrics[-1]
            val_acc = val_metrics[4]
            test_acc  = test_metrics[4]

            # Print evaluation result
            print_metrics('train', train_loss, train_metrics)
            print_metrics('valid', val_loss, val_metrics)
            print_metrics('test', test_loss, test_metrics)

            # Sava the model by auc
            if val_auc >best_val_auc:

                count = 0
                best_model = model
                best_val_acc=val_acc
                best_val_auc = val_auc
                best_val_aupr = val_aupr
                kfold_val_true=val_trues
                kfold_val_score=val_scores

                best_test_auc = test_auc
                best_test_aupr = test_aupr
                best_test_acc=test_acc
                best_test_scores=test_scores


                print("!!!Get better model with valid AUC:{:.6f}. ".format(val_auc))

                torch.save(best_model, os.path.join(save_path,
                                                'model_{}_val_acc{:.3f}_auc{:.3f}_aupr{:.3f}_test_acc{:.3f}_auc{:.3f}_aupr{:.3f}.pkl'
                                                    .format(i + 1,best_val_acc,best_val_auc,best_val_aupr,best_test_acc,
                                                                                          best_test_auc,best_test_aupr)))

            else:
                count += 1
                if count >= patience:
                    torch.save(best_model, os.path.join(save_path,
                                                        'model_{}_acc{:.3f}_auc{:.3f}.pkl'.format(i + 1, best_test_acc,
                                                                                            best_test_auc)))
                    print(f'Fold {i + 1} training done!!!\n')
                    break


        val_auprs.append(best_val_aupr)
        test_auprs.append(best_test_aupr)
        val_aucs.append(best_val_auc)
        test_aucs.append(best_test_auc)
        kfold_val_trues.append(kfold_val_true)
        kfold_val_scores.append(kfold_val_score)
        kfold_test_scores.append(best_test_scores)

    print(f'Cell line {cell_line} model training done!!!\n')

    final_test_scores = np.sum(np.array(kfold_test_scores), axis=0) / kfold

    # Cal the best threshold

    best_acc_threshold, best_acc = best_acc_thr(test_trues, final_test_scores)

    print('The best acc threshold is {:.2f} with the best acc({:.3f}).'.format(best_acc_threshold, best_acc))

    for i, (val_trues, val_scores) in enumerate(zip(kfold_val_trues, kfold_val_scores)):
        print('Fold {}:.'.format(i+1))
        val_metric=cal_metrics(val_trues,val_scores,best_acc_threshold)
        print_metrics('valid', 0, val_metric)

    # Select the best threshold by acc
    final_test_metrics = cal_metrics(test_trues, final_test_scores, best_acc_threshold)[:]
    print_metrics('Final test', test_loss, final_test_metrics)




def cal_by_epoch(mode, model, loader, loss, optimizer=None):
    # Model on train mode
    model.train() if mode == 'train' else model.eval()
    all_trues, all_scores= [],[]
    losses, sample_num = 0.0, 0
    for iter_idx, (X, y) in enumerate(loader):
        sample_num += y.size(0)

        # Create vaiables
        with torch.no_grad():
            X_var = torch.autograd.Variable(X.to(device))
            y_var = torch.autograd.Variable(y.to(device).float())

        # compute output
        model = model.to(device)
        output,att= model(X_var)
        output=output[:,1]

        # calculate and record loss
        loss_batch = loss(output.float(), y_var.float())
        losses += loss_batch.item()

        # compute gradient and do SGD step when training
        if mode == 'train':
            optimizer.zero_grad()
            loss_batch.requires_grad_(True)
            loss_batch.backward()
            optimizer.step()


        all_trues.append(y_var.data.cpu().numpy())
        all_scores.append(output.data.cpu().numpy())

    all_trues = np.concatenate(all_trues, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    return all_trues, all_scores, losses/sample_num



if __name__ == '__main__':
        train_cv()
