"""
Transformer for EEG classification
"""


import os
import numpy as np
import math
import random
import time
import scipy.io
from sklearn import preprocessing
from sklearn import model_selection

from torch.utils.data import DataLoader
from torch.autograd import Variable
# from torchsummary import summary

import torch

import torch.nn.functional as F

from torch import nn
from torch import Tensor

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp
from spatial_filter import spatial_filter
# from confusion_matrix import plot_confusion_matrix
# from cm_no_normal import plot_confusion_matrix_nn
# from torchsummary import summary

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

# writer = SummaryWriter('/home/syh/Documents/MI/code/Trans/TensorBoardX/')

# torch.cuda.set_device(6)
# gpus = [6]
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
torch.cuda.is_available()
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  

device = torch.device(dev)  

####################### DEFINE SOME GLOBAL VARIABLES ################
#  To be put in another file                                        #
#  Stuff like layer sizes, number of time channels etc.             #
n_classes = 5
n_principle_comp = 4
n_time_steps=170                                                                  
#####################################################################

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size, kc):
        # self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(1, 2, (1, kc), (1, 1)),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2, emb_size, (16, 5), stride=(1, 5)), # TODO: WHAT IS THIS 16??{num_classes * num_pcs} WHO IS THIS 5???? 
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # self.positions = nn.Parameter(torch.randn((100 + 1, emb_size)))
        # self.positions = nn.Parameter(torch.randn((2200 + 1, emb_size)))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)

        # position
        # x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size, num_heads):
        super().__init__(*[TransformerEncoderBlock(emb_size, num_heads=num_heads) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'), # the mysterious compression
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return x, out


class ViT(nn.Sequential):
    def __init__(self, n_time_steps:int,emb_size=10, depth=4, n_classes=n_classes, kc=51, num_heads=5, **kwargs):
        super().__init__(
            # channel_attention(),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(n_time_steps),
                    channel_attention(),
                    nn.Dropout(0.5),
                )
            ),

            PatchEmbedding(emb_size, kc),
            TransformerEncoder(depth, emb_size, num_heads=num_heads), # TODO: GROUP10: include heads (hyperparameter)?
            ClassificationHead(emb_size, n_classes)
        )


class channel_attention(nn.Module):
    def __init__(self, sequence_num=n_time_steps, inter=30):
        super(channel_attention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(self.sequence_num / self.inter)  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(n_classes*n_principle_comp, n_classes*n_principle_comp),
            nn.LayerNorm(n_classes*n_principle_comp),  # also may introduce improvement to a certain extent
            nn.Dropout(0.3)
        )
        self.key = nn.Sequential(
            nn.Linear(n_classes*n_principle_comp, n_classes*n_principle_comp),
            # nn.LeakyReLU(),
            nn.LayerNorm(n_classes*n_principle_comp),
            nn.Dropout(0.3)
        )

        # self.value = self.key
        self.projection = nn.Sequential(
            nn.Linear(n_classes*n_principle_comp, n_classes*n_principle_comp),
            # nn.LeakyReLU(),
            nn.LayerNorm(n_classes*n_principle_comp),
            nn.Dropout(0.3)
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        temp = rearrange(x, 'b o c s->b o s c')
        temp_query = rearrange(self.query(temp), 'b o s c -> b o c s')
        temp_key = rearrange(self.key(temp), 'b o s c -> b o c s')

        channel_query = self.pooling(temp_query)
        channel_key = self.pooling(temp_key)

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = torch.einsum('b o c s, b o m s -> b o c m', channel_query, channel_key) / scaling

        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum('b o c s, b o c m -> b o c s', x, channel_atten_score)
        '''
        projections after or before multiplying with attention score are almost the same.
        '''
        out = rearrange(out, 'b o c s -> b o s c')
        out = self.projection(out)
        out = rearrange(out, 'b o s c -> b o c s')
        return out


class Trans():
    def __init__(self, path:str, filename:str, outdir:str, 
        slice_size=10, h=5, kc=51,
        batch_size=50, n_epochs=1000, c_dim=4,
        lr=0.0002,b1=0.5,b2=0.9):
        '''__init__ - initialization for the Trans class
        Necessary Inputs:
            path (str) - path to the data
            filename (str) - name for the datafile
            outdir (str) - path to output directory
        Hyperparameters:
            slice_size - number of slices for attention mechanism
            h - number of heads
            c_dim - size of convolution # TODO: is this working???
        '''
        super(Trans, self).__init__()
        assert filename != '' # cannot be empty
        assert path != ''
        assert outdir != ''
        self.path = path # path to datafile
        self.filename = filename # include whole path to file here as well
        self.outdir = outdir #output directory
        self.batch_size = batch_size # size of batch
        self.n_epochs = n_epochs # number of epochs
        # HyperParameters
        self.heads = h # number of heads
        self.slice_size = slice_size # number of slices for attention
        self.kc = kc # convolution filter size
        # self.n_Ceeg = None # number of Ceeg channels
        # self.n_time_steps = 170 # time series size
        # self.channels = channels # TODO: remove me?
        self.c_dim = c_dim   # convolution dimension?
        self.lr = lr # learning rate
        self.b1 = b1 # optimization parameter
        self.b2 = b2 # optimization parameter
        # self.start_epoch = start_epoch
        self.root = '...'  # the path of data # change this?

        self.pretrain = False

        # self.log_write = open(os.path.join(self.outdir,f"log_{filename[:-4]}.txt"), "w") # TODO: CHANGE ME

        # self.img_shape = (self.img_height, self.img_width) # input image size

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        # self.Tensor = torch.FloatTensor
        # self.LongTensor = torch.LongTensor
        self.criterion_l1 = torch.nn.L1Loss().to(device)
        self.criterion_l2 = torch.nn.MSELoss().to(device)
        self.criterion_cls = torch.nn.CrossEntropyLoss().to(device)

        self.get_source_data()

        self.model = ViT(n_time_steps=self.n_time_steps, kc=self.kc, num_heads=self.heads).to(device) # TODO: GROUP10 include the number of classes here
        # self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.to(device)
        # summary(self.model, (1, 16, 1000))
        print("HERE BE SUMMARY")

        self.centers = {}

    # returns NONE
    # pulls data from the datafile in this object's fields
    # also sets image height and width
    def get_source_data(self):

        # to get the data of target subject
        self.total_data = np.load(os.path.join(self.path,self.filename), allow_pickle=True)
        
        self.train_data = self.total_data.item()['x_train'].transpose(0,2,1) # GROUP10; transposed to form n x Ceeg x T
        self.train_labels = self.total_data.item()['y_train'] # GROUP10
        self.test_data = self.total_data.item()['x_test'].transpose(0,2,1) #GROUP10; transposed to form n x Ceeg x T
        self.test_labels = self.total_data.item()['y_test'] #GROUP10

        _, self.n_Ceeg, self.n_time_steps = self.train_data.shape
        return self.train_data, self.train_labels, self.test_data, self.test_labels

    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Do some data augmentation is a potential way to improve the generalization ability
    def aug(self, img, label):
        aug_data = []
        aug_label = []
        return aug_data, aug_label

    # preprocessing for each validation split
    # Input: X - the (training) data
    #        y - the labels
    # Returns:
    #   mu - the average over the training 
    #   sigma - the standard deviation over the training
    #   W (tuple) - the spatial filter weights
    def preprocessing(self, X, y):
        assert X.shape[0] == y.shape[0] # same number of examples
        mu = np.average(X)
        sigma = np.std(X)
        W = spatial_filter(X, y)
        return mu, sigma, W.T

    '''crossVal - perform k-fold crossvalidation
    Input: num_folds (int) - the number of folds for cross validation
    Outputs:
        avg_accs - the average k-fold crossvalidation accuracies
        best_accs - the best accuracy on each fold
    '''
    def crossVal(self, num_folds:int): # NOTE: must run getSourceData before this method! Assume that it has been called?
        print("Initiate cross-validation.")
        #self.get_source_data()
        ## formatting for splits found at https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
        kf = model_selection.KFold(n_splits=num_folds, random_state=42, shuffle=True) # TODO: randomize random state?
        kf.get_n_splits(self.train_data)
        fold_num = 0 # number for this current fold
        running_total_acc = 0 # used in calculating average accuracy
        bestAccs = []
        averAccs = []
        Y_trues = []
        Y_preds = []
        for train_idxs, val_idxs in kf.split(self.train_data):
            print("Fold number %d", fold_num)
            # self.log.write(f"Fold number {fold_num}")
            # Generate train and validation splits #
            X_t = self.train_data[train_idxs]   # train split
            y_t = self.train_labels[train_idxs] # train labels
            X_v = self.train_data[val_idxs]     # validation split
            y_v = self.train_labels[val_idxs]   # validation labels
            # Generate filtered versions on the z-scored
            mu, sigma, W = self.preprocessing(X_t, y_t)
            
            
            X_t = np.expand_dims((X_t - mu) / sigma, axis=1) # add channel dimension for convolution
            X_v = np.expand_dims((X_v - mu) / sigma, axis=1)
            print(f"X_t shape before dot product:{X_t.shape}")
            print(f"X_v shape before dot product:{X_v.shape}")     
            # do dot product
            print(f"W shape: {W.shape}")
            X_t = np.einsum('abcd, ce -> abed', X_t, W)
            X_v = np.einsum('abcd, ce -> abed', X_v, W) # TODO: consider einsum?

            
            print(f"W dot X_t shape :{X_t.shape}")
            print(f"W dot X_v shape:{X_v.shape}")
            _, tcounts = np.unique(y_t, return_counts=True)
            _, vcounts = np.unique(y_v, return_counts=True)
            print(f"y_t counts:{tcounts}")
            print(f"y_v counts:{vcounts}")
            
            # train model on training set and predict accuracy on validation set #
            bestAcc, averAcc, Y_true, Y_pred = self.train(X_t, y_t, X_v, y_v) # edit train method to no longer use the self.parameters
            # pred, acc = self.classify(X_v, y_v) # TODO: write this method # Is this necessary if testing on validation set can be done in self.train?
            bestAccs.append(bestAcc) # best accuracy in the train method
            averAccs.append(averAcc) # average accuracy from the train method
            Y_trues.append(Y_true)
            Y_preds.append(Y_pred)
            # update accuracy statistics #
            # running_total_acc += acc
            # if best_acc < acc: best_acc = acc
            # print(f"\tValidation accuracy: {acc}\n\tBest acc so far: {best_acc}")
            # self.log.write(f"\tValidation accuracy: {acc}\n\tBest acc so far: {best_acc}")
            fold_num += 1
        # avg_acc = running_total_acc / num_folds
        return averAccs, bestAccs
        # Proc: save statistics and calculate the avg. performance (what objective function should we use for validation?)
        # split HERE # split training into [train, validation]
        # for fold in numfolds: # iterate over all train-validation splits
        #     self.preprocessing(xtrain,ytrain) # preprocess the data
            # apply preprocessing to validation set
            # train the network
            # apply on validation set
            # save statistics

    def train(self, train_data, train_label, val_data, val_label): # TODO: edit to take in the train data and labels
        # TODO: FIX THIS METHOD DRASTICALLY
        # GROUP10:
        # both the train and validation sets are passed into this function to allow tracking of potential overfitting
        # Note: It might be worthwhile to include a mode where val_data and val_label are empty and only statistics on
        #       fit to training set are recorded.

        # img, label, test_data, test_label = self.get_source_data()
        img = train_data
        label = train_label
        test_data = val_data # validation set is misnomered, per original authors
        test_label = val_label
        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)


        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        # Train the model
        total_step = len(self.dataloader)
        curr_lr = self.lr
        # some better optimization strategy is worthy to explore. Sometimes terrible over-fitting.


        for e in range(self.n_epochs):
            in_epoch = time.time()
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):

                img = Variable(img.to(device).type(self.Tensor))
                # print(f"img shape: {img.shape}")
                label = Variable(label.to(device).type(self.LongTensor))
                tok, outputs = self.model(img)
                # print(f"output size: {outputs.shape}")
                loss = self.criterion_cls(outputs, label)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            out_epoch = time.time()

            if (e + 1) % 1 == 0: # TODO: report step size
                self.model.eval()
                Tok, Cls = self.model(test_data)

                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                # print(f"predicted labels: {y_pred}")
                # print(f"test_label: {test_label}")
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
                print('Epoch:', e,
                      '  Train loss:', loss.detach().cpu().numpy(),
                      '  Validation loss:', loss_test.detach().cpu().numpy(),
                      '  Train accuracy:', train_acc,
                      '  Validation accuracy is:', acc)
                # self.log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred

        #torch.save(self.model.module.state_dict(), 'model.pth')
        averAcc = averAcc / num
        print('The average epoch accuracy is:', averAcc)
        print('The best epoch accuracy is:', bestAcc)
        # self.log_write.write('The average epoch accuracy is: ' + str(averAcc) + "\n")
        # self.log_write.write('The best epoch accuracy is: ' + str(bestAcc) + "\n")

        return bestAcc, averAcc, Y_true, Y_pred # TODO: edit what is being returned


def main():
    best = 0
    aver = 0
    result_write = open(r"..\..\output\sub_result.txt", "w") # TODO: EDIT PATH

    #PATH = ''
    DATADIR = os.path.join('..','..','output')#r'..\..\output'
    OUTDIR = os.path.join('..','..','output')#r'..\..\output'
    FILENAME = r'saved_data.npy'
    # for i in range(9): # TODO: change for loop? are they iterating over files?
    # for file in dir(PATHTODATA\\\) # for file in data directory that we want to classify on
    seed_n = np.random.randint(500)
    print('seed is ' + str(seed_n))
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    print(f'File {FILENAME}')
    trans = Trans(DATADIR, FILENAME, outdir=OUTDIR)

    # get the data and start the training process
    trans.get_source_data()
    bestAcc, averAcc, Y_true, Y_pred = trans.crossVal(10)

    print('THE BEST ACCURACY IS ' + str(bestAcc))
    print(f"Averac: {averAcc}")
    result_write.write('File ' + FILENAME + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
    # result_write.write('**File ' + FILENAME + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
    # result_write.write('File ' + FILENAME + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")
    # plot_confusion_matrix(Y_true, Y_pred, i+1)
    best = best + bestAcc
    aver = aver + averAcc
    # if i == 0:
    #     yt = Y_true
    #     yp = Y_pred
    # else:
    #     yt = torch.cat((yt, Y_true)) # GROUP10: why cat?
    #     yp = torch.cat((yp, Y_pred))
    yt = Y_true
    yp = Y_pred

    # best = best / 9
    # aver = aver / 9
    # plot_confusion_matrix(yt, yp, 666)
    result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()


if __name__ == "__main__":
    main()