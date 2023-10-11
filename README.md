# CellCircLoc 

Deep Neural Network for Predicting and Explaining Cell line-specific CircRNA Subcellular Localization 

The web server for prediction and visualization available at [http://csuligroup.com:8000/cellcircloc](http://csuligroup.com:8000/cellcircloc)


## Requirements

- python=3.7.0
- numpy=1.19.2
- scikit-learn=0.24.2
- pytorch=1.9.0


## Usage

Create and activate the environment.You can train the model in a very simple way

    python main.py


To change model hyperparameters, You can do like this:

    python main.py --epoch_num 1000

Or you can change the hyperparameters directly in code.
The hyperparameters are then introduced


#### Specify model hyperparameters

    seed, cell_line, emb_type, max_len, epoch_num, batch_size, patience, threshold, save_path,lr,weight_decay,\
    cnn_kernel_size,lstm_hidden_dim,lstm_num_layers,lstm_dropout,l_dropout,\
    m, h, d_ff,act_num_of_hid,n_list,repeat_num,act_dropout=\
    2023,"K562","onehot",2000,1000,64,100,0.5,os.getcwd(),0.001,0.0,\
    3,8,1,0.0,0.0,\
    100,6,6,6,[3],12,0.0


seed: the random seed

cell_line: show the training cell line. To change the cell line,please replace the 'data.pkl' in main directory

max_len: maximum nucleotide length. Those exceeding the maximum length will be discarded

epoch_num: num of epoch

batch_size: batch size

patience: early stop strategy with a patience

threshold: decide each nucleotide to be label 0 or 1

save_path: save path

lr: learning rate

weight_decay: weight decay

cnn_kernel_size: kernel size of convolutional neural network layer

lstm_hidden_dim: hidden layer dim of Bi-LSTM

lstm_num_layers: Bi-LSTM num of layers

lstm_dropout: dropout of Bi-LSTM

l_dropout: Linear dropout

m: num of convolutional filters in attentive convolution Transformer blocks

h: num of heads in attentive convolution Transformer blocks

d_ff: dim of FeedForwardNetwork in attentive convolution Transformer blocks

act_num_of_hid: hidden dim of attentive convolution Transformer blocks

n_list: kernel size of convolutional filters in attentive convolution Transformer blocks

repeat_num: repeated blocks of each kernel size in attentive convolution Transformer blocks

act_dropout: dropout in attentive convolution Transformer blocks



#### Prediction

To make prediction,you can change the code to change testset into what you want to predict,here is a simple example of predict one nucleotide

    test_dataset = CelllineDataset(['ATTTCTTGGGGGGGGGGGCCC'],np.array(1),emb_type=emb_type,max_len=max_len)


#### data	

The data of seven cell line is in fold 'data of all cell lines'.


There are seven cell lines:
K562, HepG2, Hela-S3, HUVEC, keratinocyte,
GM12878, H1-hESC


If you want to change the cell line, replace the 'data.pkl' in main directory.


## License

This project is licensed under the MIT License - see the LICENSE.txt file for details



## Citation

Min Zeng, Jingwei Lu, Yiming Li, Chengqian Lu, Shichao Kan, Fei Guo, Min Li*,"CellCircLoc: Deep Neural Network for Predicting and Explaining Cell line-specific CircRNA Subcellular Localization"

  
  
  
  
