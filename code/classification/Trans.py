"""
Transformer for EEG classification
"""


import os
import numpy as np
import math
import random
import time
import scipy.io

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


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size):
        # self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(1, 2, (1, 51), (1, 1)), # GROUP10: why these values?
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2, emb_size, (16, 5), stride=(1, 5)),
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
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return x, out


class ViT(nn.Sequential):
    def __init__(self, emb_size=10, depth=3, n_classes=4, **kwargs):
        super().__init__(
            # channel_attention(),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(1000),
                    channel_attention(),
                    nn.Dropout(0.5),
                )
            ),

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )


class channel_attention(nn.Module):
    def __init__(self, sequence_num=1000, inter=30):
        super(channel_attention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(self.sequence_num / self.inter)  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(16, 16),
            nn.LayerNorm(16),  # also may introduce improvement to a certain extent
            nn.Dropout(0.3)
        )
        self.key = nn.Sequential(
            nn.Linear(16, 16),
            # nn.LeakyReLU(),
            nn.LayerNorm(16),
            nn.Dropout(0.3)
        )

        # self.value = self.key
        self.projection = nn.Sequential(
            nn.Linear(16, 16),
            # nn.LeakyReLU(),
            nn.LayerNorm(16),
            nn.Dropout(0.3),
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
    def __init__(self, path:str, filename:str, outdir:str, batch_size=50,n_epochs=2000,img_height=21,img_width=850,c_dim=4,lr=0.0002,b1=0.5,b2=0.9):
        super(Trans, self).__init__()
        assert filename != '' # cannot be empty
        self.path = path # path to datafile
        self.filename = filename # include whole path to file here as well
        self.outdir = outdir #output directory
        self.batch_size = batch_size # size of batch
        self.n_epochs = n_epochs # number of epochs
        self.img_height = img_height # number of channels
        self.img_width = img_width # time series size?
        # self.channels = channels # TODO: remove me?
        self.c_dim = c_dim   # convolution dimension?
        self.lr = lr # learning rate
        self.b1 = b1 # optimization parameter
        self.b2 = b2 # optimization parameter
        # self.start_epoch = start_epoch
        self.root = '...'  # the path of data # change this?

        self.pretrain = False

        self.log_write = open(os.path.join(self.outdir,f"log_{filename[:-4]}.txt"), "w") # TODO: CHANGE ME

        self.img_shape = (self.img_height, self.img_width) # input image size

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().to(device)
        self.criterion_l2 = torch.nn.MSELoss().to(device)
        self.criterion_cls = torch.nn.CrossEntropyLoss().to(device)

        self.model = ViT().to(device)
        # self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.to(device)
        # summary(self.model, (1, 16, 1000))
        print("HERE BE SUMMARY")

        self.centers = {}

    # returns NONE
    def get_source_data(self):

        # to get the data of target subject
        self.total_data = np.load(os.path.join(self.path,self.filename), allow_pickle=True)
        self.train_data = self.total_data.item()['x_train'].transpose(0,2,1) # GROUP10; transposed to form n x Ceeg x T
        self.train_label = self.total_data.item()['y_train'] # GROUP10
        self.test_data = self.total_data.item()['x_test'].transpose(0,2,1) #GROUP10; transposed to form n x Ceeg x T
        self.test_label = self.total_data.item()['y_test'] #GROUP10
        return self.train_data, self.train_label, self.test_data, self.test_label
        # below is a lot of stuff we don't need.
        # print(self.train_data.shape, self.train_label.shape)
        # print(self.test_data.shape, self.test_label.shape)
        # GROUP10
        ## z-score standardization
        # self.train_data = 
        # END GROUP10
        
        # START NOT US
        # self.train_data = np.transpose(self.train_data, (2, 1, 0)) #OG
        # self.train_data = np.expand_dims(self.train_data, axis=1) #OG
        # self.train_label = np.transpose(self.train_label) #OG

        # self.allData = self.train_data #OG
        # self.allLabel = self.train_label[0] #OG

        # test data
        # to get the data of target subject
        # self.test_tmp = scipy.io.loadmat(self.root + 'A0%dE.mat' % self.nSub)
        # self.test_data = self.total_data.item()['x_test']
        # self.test_label = self.total_data.item()['y_test']

        # self.train_data = self.train_data[250:1000, :, :]
        # self.test_data = np.transpose(self.test_data, (2, 1, 0))
        # self.test_data = np.expand_dims(self.test_data, axis=1)
        # self.test_label = np.transpose(self.test_label)

        # self.testData = self.test_data
        # self.testLabel = self.test_label

        # Mix the train and test data - a quick way to get start
        # But I agree, just shuffle data is a bad measure
        # You could choose cross validation, or get more data from more subjects, then Leave one subject out
        # all_data = np.concatenate((self.allData, self.testData), 0) # what is this concat???
        # all_label = np.concatenate((self.allLabel, self.testLabel), 0)
        # all_shuff_num = np.random.permutation(len(all_data))
        # all_data = all_data[all_shuff_num]
        # all_label = all_label[all_shuff_num]
        # END NOT US

        # self.allData = all_data[:641]
        # self.allLabel = all_label[:641]
        # self.testData = all_data[641:] ##### Group10 note: the number 641 here is chosen based on test, train split ####
        # self.testLabel = all_label[641:] ##TODO: make it a floor function

        ### TODO: GROUP10
        #  PROBABLY BETTER NOT TO USE ANY OF THE CODE ABOVE SINCE WE ALREADY CREATED 
        # TRAIN TEST SPLIT. THE ONLY PART WORTH KEEPING MIGHT BE THE CONCATANATING ALL THE DATA###

        # standardize # DON"T STANDARDIZE HERE. STANDARDIZE IN training
        # target_mean = np.mean(self.allData)
        # target_std = np.std(self.allData)
        # self.allData = (self.allData - target_mean) / target_std
        # self.testData = (self.testData - target_mean) / target_std

        # tmp_alldata = np.transpose(np.squeeze(self.allData), (0, 2, 1))
        # Wb = spatial_filter(tmp_alldata, self.allLabel-1)  # common spatial pattern
        # self.allData = np.einsum('abcd, ce -> abed', self.allData, Wb)
        # self.testData = np.einsum('abcd, ce -> abed', self.testData, Wb)
        # return self.allData, self.allLabel, self.testData, self.testLabel

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
    #   X_out - the standardized training data
    #   mu - the average over the training 
    #   sigma - the standard deviation over the training
    #   W (tuple) - the spatial filter weights
    def preprocessing(self, X, y):
        assert X.shape[0] == y.shape[0] # same number of examples
        mu = np.average(X)
        sigma = np.std(X)
        W = spatial_filter(X, y)
        X_out = (X - mu) / sigma
        return X_out, mu, sigma, W

    def train(self):


        img, label, test_data, test_label = self.get_source_data()
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

        # Train the cnn model
        total_step = len(self.dataloader)
        curr_lr = self.lr
        # some better optimization strategy is worthy to explore. Sometimes terrible over-fitting.


        for e in range(self.n_epochs):
            in_epoch = time.time()
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):

                img = Variable(img.to(device).type(self.Tensor))
                label = Variable(label.to(device).type(self.LongTensor))
                tok, outputs = self.model(img)
                loss = self.criterion_cls(outputs, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            out_epoch = time.time()

            if (e + 1) % 1 == 0:
                self.model.eval()
                Tok, Cls = self.model(test_data)

                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
                print('Epoch:', e,
                      '  Train loss:', loss.detach().cpu().numpy(),
                      '  Test loss:', loss_test.detach().cpu().numpy(),
                      '  Train accuracy:', train_acc,
                      '  Test accuracy is:', acc)
                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred

        torch.save(self.model.module.state_dict(), 'model.pth')
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        return bestAcc, averAcc, Y_true, Y_pred


def main():
    best = 0
    aver = 0
    result_write = open(r"D:\thewi\Documents\UM\WN22\ML\Project\Datasets\ml-project\output\sub_result.txt", "w") # TODO: EDIT PATH

    #PATH = ''
    DATADIR = r'D:\thewi\Documents\UM\WN22\ML\Project\Datasets\ml-project\output'
    OUTDIR = r'D:\thewi\Documents\UM\WN22\ML\Project\Datasets\ml-project\output'
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
    bestAcc, averAcc, Y_true, Y_pred = trans.train()
    print('THE BEST ACCURACY IS ' + str(bestAcc))
    result_write.write('File ' + FILENAME + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
    result_write.write('**File ' + FILENAME + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
    result_write.write('File ' + FILENAME + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")
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

    best = best / 9
    aver = aver / 9
    # plot_confusion_matrix(yt, yp, 666)
    result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()


if __name__ == "__main__":
    main()