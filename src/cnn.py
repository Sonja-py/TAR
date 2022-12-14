# -*- coding: utf-8 -*-
"""Copy of CNN ASR Final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MyJQp6HI02sbarn75hJ2MSD5kYImvtfT
"""

# Install conda and add channels to look for packages in
import sys
# ! wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
# ! chmod +x Anaconda3-2022.10-Linux-x86_64.sh
# ! bash ./Anaconda3-2022.10-Linux-x86_64.sh -b -f -p /usr/local
# sys.path.append('/usr/local/lib/python3.8/site-packages/')
# ! conda update -n base -c defaults conda -y
# ! conda config --add channels bioconda
# ! conda config --add channels conda-forge

# Commented out IPython magic to ensure Python compatibility.
# %pip install torch

# Commented out IPython magic to ensure Python compatibility.
# %pip install torchaudio

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
#!pip install torchaudio==0.13.0    # This will force you to restart your kernel. That's ok

"""**One Hot Encoding for Text**"""

class TextTransform:
  def __init__(self):
    self.char_dict = {}
    self.idx_dict = {}
    for i in range(26):
      self.char_dict[chr(97+i)] = i
      self.idx_dict[i] = chr(97+i)
    self.char_dict[' '] = 26
    self.idx_dict[26] = ' '
    self.char_dict[chr(39)] = 27
    self.idx_dict[27] = chr(39)

  def text_to_int(self, text):
    int_seq = []
    for char in text:
        int_seq.append(self.char_dict[char])
    return int_seq

  def int_to_text(self, int_seq):
    text = []
    for num in int_seq:
      text.append(self.idx_dict[num])
    return ''.join(text)

text_transform = TextTransform()

train_transforms = nn.Sequential(
  torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
  torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
  torchaudio.transforms.TimeMasking(time_mask_param=100))

test_transforms = torchaudio.transforms.MelSpectrogram()


def process_libri(data, data_type):
  spectrograms = []
  labels = []
  input_len = []
  label_len = []
  for item in data:
    if data_type == 'train':
        spec = train_transforms(item[0]).squeeze(0).transpose(0, 1)
    else:
        spec = test_transforms(item[0]).squeeze(0).transpose(0, 1)
    spectrograms.append(spec)
    label = torch.Tensor(text_transform.text_to_int(item[2].lower()))
    labels.append(label)
    input_len.append(spec.shape[0]//2)
    label_len.append(len(label))

  spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
  labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

  return spectrograms, labels, input_len, label_len


def process_l2arctic(data, data_type):
  spectrograms = []
  labels = []
  input_len = []
  label_len = []
  for item in data:
    if data_type == 'train':
        spec = train_transforms(item[0]).squeeze(0).transpose(0, 1)
    else:
        spec = test_transforms(item[0]).squeeze(0).transpose(0, 1)
    spectrograms.append(spec)
    label = torch.Tensor(item[1])
    labels.append(label)
    input_len.append(spec.shape[0]//2)
    label_len.append(len(label))

  spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
  labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

  return spectrograms, labels, input_len, label_len

"""**Accuracy**"""

def decoder(output, labels, label_lengths, collapse_repeated=True):
  arg_maxes = torch.argmax(output, dim=2)
  output = []
  targets = []
  for i, args in enumerate(arg_maxes):
    decode = []
    targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
    for j, index in enumerate(args):
      if index != 28:
        if collapse_repeated and j != 0 and index == args[j -1]:
          continue
        decode.append(index.item())
    output.append(text_transform.int_to_text(decode))
  return output, targets

def wer(reference, hypothesis):
  r = reference.split()
  h = hypothesis.split()
  # costs will holds the costs, as in the Levenshtein distance algorithm
  costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
  # backtrace will hold the operations we've done.
  # so we could later backtrace, like the WER algorithm requires us to.
  backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
 
  OP_OK = 0
  OP_SUB = 1
  OP_INS = 2
  OP_DEL = 3
  DEL_PENALTY = 1
  INS_PENALTY = 1
  SUB_PENALTY = 1
    
  # First column represents the case where we achieve zero
  # hypothesis words by deleting all reference words.
  for i in range(1, len(r)+1):
      costs[i][0] = DEL_PENALTY*i
      backtrace[i][0] = OP_DEL
  
  # First row represents the case where we achieve the hypothesis
  # by inserting all hypothesis words into a zero-length reference.
  for j in range(1, len(h) + 1):
      costs[0][j] = INS_PENALTY * j
      backtrace[0][j] = OP_INS
    
  # computation
  for i in range(1, len(r)+1):
      for j in range(1, len(h)+1):
          if r[i-1] == h[j-1]:
              costs[i][j] = costs[i-1][j-1]
              backtrace[i][j] = OP_OK
          else:
              substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
              insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
              deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1
                
              costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
              if costs[i][j] == substitutionCost:
                  backtrace[i][j] = OP_SUB
              elif costs[i][j] == insertionCost:
                  backtrace[i][j] = OP_INS
              else:
                  backtrace[i][j] = OP_DEL
                 
  # back trace though the best route:
  i = len(r)
  j = len(h)
  numSub = 0
  numDel = 0
  numIns = 0
  numCor = 0
  while i > 0 or j > 0:
      if backtrace[i][j] == OP_OK:
          numCor += 1
          i-=1
          j-=1
      elif backtrace[i][j] == OP_SUB:
          numSub +=1
          i-=1
          j-=1
      elif backtrace[i][j] == OP_INS:
          numIns += 1
          j-=1
      elif backtrace[i][j] == OP_DEL:
          numDel += 1
          i-=1
  return (numSub + numDel + numIns) / (float) (len(r))

"""**Loading L2Arctic Data**"""

# from google.colab import drive
# drive.mount('/content/drive')

def load_l2arctic():

    # Returns 2D array: speaker, audio
    def load_audio():
      '''
      Reducing for only for one speaker
      audio = [0]*17
      for speaker in range(17):
        print(speaker)
      '''
      speaker = 0
      files = os.listdir('../l2arctic_recordings/'+str(speaker))
      files.sort()
      audio = [0]*1131
#      individual_audio = [0]*1131
      i=0
      for file in files:
        waveform, sample_rate = torchaudio.load('../l2arctic_recordings/'+str(speaker)+'/'+file)
        
        # Match sampling rate between corpuses
        resample = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resample(waveform)
        audio[i] = waveform
#        individual_audio[i] = waveform
        i+=1
#      audio[speaker] = individual_audio
      return audio
    
    # Returns array of tensors for each transcript
    def load_transcript():
#      transcript_sentences = [""]*1131
      transcript_words = [0]*1131
      arctic_f = open('../arctic_transcript.log', 'r')
      i = 0
      for line in arctic_f:
        line = line.strip().lower()
#        transcript_sentences[i] = line
        transcript_words[i] = text_transform.text_to_int(line)
        i+=1
      arctic_f.close()
      return transcript_words

    print('loading audio...')
    audio = load_audio()
    print('loading transcript...')
    transcript = load_transcript()

    #return audio, transcript

    # Separate into training/testing data sets
    print('partitioning...')
    train_data = [0]*904
    test_data = [0]*227
    train_idx = 0
    test_idx = 0
    for i in range(len(audio)):
      if (i%5 == 0):    # 80:20 train:test split
        test_data[test_idx] = (audio[i], transcript[i%1131])
        test_idx+=1
      else:
        train_data[train_idx] = (audio[i], transcript[i%1131])
        train_idx+=1

    return train_data, test_data

"""**Create the model**"""

class CNNLayerNorm(nn.Module):
  def __init__(self, nfeats):
    super(CNNLayerNorm, self).__init__()
    self.layer_norm = nn.LayerNorm(nfeats)

  def forward(self, x):
    # x (batch, channel, feature, time)
    x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
    x = self.layer_norm(x)
    return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 


class ResidualCNN(nn.Module):
  def __init__(self, in_channels, out_channels, kernel, stride, dropout, nfeats):
    super(ResidualCNN, self).__init__()

    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
    self.dropout = nn.Dropout(dropout)
    self.layer_norm1 = CNNLayerNorm(nfeats)
    self.layer_norm2 = CNNLayerNorm(nfeats)

  def forward(self, x):
    residual = x  # (batch, channel, feature, time)
    x = self.layer_norm1(x)
    x = F.relu(x)
    x = self.dropout(x)
    x = self.conv1(x)
    x = self.layer_norm2(x)
    x = F.relu(x)
    x = self.dropout(x)
    x = self.conv2(x)
    x += residual
    return x # (batch, channel, feature, time)


class BiRNN(nn.Module):

  def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
    super(BiRNN, self).__init__()

    self.BiRNN = nn.GRU(
        input_size=rnn_dim, hidden_size=hidden_size,
        num_layers=1, batch_first=batch_first, bidirectional=True)
    self.layer_norm = nn.LayerNorm(rnn_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = self.layer_norm(x)
    x = F.gelu(x)
    x, _ = self.BiRNN(x)
    x = self.dropout(x)
    return x


class ASR(nn.Module):
    
  def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, nfeats, stride=2, dropout=0.1):
    super(ASR, self).__init__()
    nfeats = nfeats//2
    self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

    # n residual cnn layers
    self.rescnn_layers = nn.Sequential(*[
        ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, nfeats=nfeats) 
        for _ in range(n_cnn_layers)
    ])
    self.fully_connected = nn.Linear(nfeats*32, rnn_dim)
    self.birnn_layers = nn.Sequential(*[
        BiRNN(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
              hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
        for i in range(n_rnn_layers)
    ])
    self.classifier = nn.Sequential(
        nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(rnn_dim, n_class)
    )

  def forward(self, x):
    x = self.cnn(x)
    x = self.rescnn_layers(x)
    sizes = x.size()
    x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
    x = x.transpose(1, 2) # (batch, time, feature)
    x = self.fully_connected(x)
    x = self.birnn_layers(x)
    x = self.classifier(x)
    x = F.log_softmax(x, dim=2)
    return x

"""**Training/Testing Model**"""

def train(model, device, train_loader, criterion, optimizer, scheduler, epoch):
  model.train()
  data_len = len(train_loader.dataset)
  for batch_idx, data in enumerate(train_loader):
    spectrograms, labels, input_lengths, label_lengths = data 
    spectrograms, labels = spectrograms.to(device), labels.to(device)


    output = model(spectrograms)  # (batch, time, n_class)
    output = output.transpose(0, 1) # (time, batch, n_class)
    optimizer.zero_grad()

    loss = criterion(output, labels, input_lengths, label_lengths)
    loss.backward()

    optimizer.step()
    scheduler.step()
    if batch_idx % 100 == 0 or batch_idx == data_len:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(spectrograms), data_len,
            100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, criterion, epoch):
  model.eval()
  test_loss = 0
  test_wer = []
  with torch.no_grad():
    for i, data in enumerate(test_loader):
      spectrograms, labels, input_lengths, label_lengths = data 
      spectrograms, labels = spectrograms.to(device), labels.to(device)

      output = model(spectrograms)  # (batch, time, n_class)
      output = output.transpose(0, 1) # (time, batch, n_class)

      loss = criterion(output, labels, input_lengths, label_lengths)
      test_loss += loss.item() / len(test_loader)

      decoded_preds, decoded_targets = decoder(output.transpose(0, 1), labels, label_lengths)
      for j in range(len(decoded_preds)):
          test_wer.append(wer(decoded_targets[j], decoded_preds[j]))
          
  avg_wer = sum(test_wer)/len(test_wer)
  print('Test set: Average loss: {:.4f}, Average WER: {:.4f}\n'.format(test_loss, avg_wer))
  return test_loss, avg_wer


def main(learning_rate, batch_size, epochs, train_dataset, test_dataset, optname):
  hparams = {
      "n_cnn_layers": 8,
      "n_rnn_layers": 3,
      "rnn_dim": 1024,
      "n_class": 29,
      "n_feats": 128,
      "stride": 2,
      "dropout": 0.6,
      "learning_rate": learning_rate,
      "batch_size": batch_size,
      "epochs": epochs
  }

  use_cuda = torch.cuda.is_available()
# torch.manual_seed(7)
  device = torch.device("cuda")

  kwargs = {'num_workers': 0, 'pin_memory': True}
  train_loader = DataLoader(dataset=train_dataset,
                            batch_size=hparams['batch_size'],
                            shuffle=True,
                            collate_fn=lambda x: process_l2arctic(x, 'train'),#change to process_libri if want libri data
                            **kwargs)
  test_loader = DataLoader(dataset=test_dataset,
                            batch_size=hparams['batch_size'],
                            shuffle=False,
                            collate_fn=lambda x: process_l2arctic(x, 'test'),
                            **kwargs)

  model = ASR(
      hparams['n_cnn_layers'], hparams['n_rnn_layers'], 
      hparams['rnn_dim'], hparams['n_class'], hparams['n_feats'], 
      hparams['stride'], hparams['dropout']).to(device)

#    print(model)
#    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

  optimizer = optim.Adam(model.parameters(), hparams['learning_rate'])
  #optimizer = optname(model.parameters(), hparams['learning_rate'])
  criterion = nn.CTCLoss(blank=28).to(device)
  scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'], 
                                          steps_per_epoch=int(len(train_loader)),
                                          epochs=hparams['epochs'],
                                          anneal_strategy='linear')
    
  loss_overall = [0]*epochs
  wer_overall = [0]*epochs

  for epoch in range(1, epochs + 1):
      train(model, device, train_loader, criterion, optimizer, scheduler, epoch)
      epoch_loss, epoch_wer = test(model, device, test_loader, criterion, epoch)
      loss_overall[epoch-1] = epoch_loss
      wer_overall[epoch-1] = epoch_wer
    
  return loss_overall, wer_overall

'''
# Load Librispeech corpus
train_dataset_full = torchaudio.datasets.LIBRISPEECH('./', url='train-clean-100', download=True)
test_dataset_full = torchaudio.datasets.LIBRISPEECH("./", url="test-clean", download=True)

# Reduce the amount of data so we can actually run it
train_dataset = [0]*(28540//10)
idx=0
i=0
for item in train_dataset_full:
  if i%10==0:
    train_dataset[idx] = item
    idx+=1
  i+=1

test_dataset = [0]*(2620//10)
idx=0
i=0
for item in test_dataset_full:
  if i%10==0:
    test_dataset[idx] = item
    idx+=1
  i+=1
'''

if __name__ == "__main__":
# Load L2Arctic corpus
    train_dataset, test_dataset = load_l2arctic()
#
#     """**Run the Model!**
#     <div>
#     Saving results as global variables so you can plot in different ways below
#     """
#
#     #learning_rate = 5e-4
    learning_rate = [.001]
    accs = []
    batch_size = 12
    epochs = 300
    optname = 'optim.AdamW'#not currently in use
    for l in learning_rate:
      loss_overall, wer_overall = main(l, batch_size, epochs, train_dataset, test_dataset, optname)
      accs.append(wer_overall)
    print(accs)
#     wer_overall = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9926263408642263, 0.9991100431629065, 1.0067662391450938, 1.0384014908243984, 1.098181969767873, 1.1995980646861697, 1.7151838372102688, 1.2487571899466179, 1.7985725239029198, 1.9890587386182086, 0.9794761650488519, 0.9349285565144597, 0.9317676816575495, 0.9115647568290738, 1.0288110690093069, 1.700100340188446, 2.0367983216000836, 0.9301928736950759, 0.9986777172900511, 1.4250738248535604, 1.0379988587037041, 0.851403870236469, 0.8327711813394633, 0.9469124121106496, 0.8637097361766961, 0.8404107762037272, 0.8106268425651687, 0.8240627091948671, 0.8077260600388348, 0.7896480294057384, 0.7855221014912642, 0.8256467956908484, 0.7961340666186476, 0.8402175895567966, 0.8696582785789836, 0.7723293960078098, 0.8058152077975861, 0.7702130101909835, 0.7841876185488516, 0.7663734454483353, 0.866645701337331, 0.7593515515321678, 0.7644300633287416, 0.7653651292417812, 0.7582001087507696, 0.7516072522680453, 0.7475337315645685, 0.7487695002012181, 0.7576297906694379, 0.745750370530106, 0.7775036515124617, 0.7504198933714346, 0.7307560158881743, 0.7791622107701407, 0.7889022993648539, 0.7517750971495463, 0.9829014719983882, 0.9169329446862482, 0.8432367436772721, 0.8417410660802728, 0.8566805807995228, 0.8665662775794934, 0.8121863696753565, 0.7860461247025123, 0.7848321037748344, 0.7484037010468728, 0.7723792971590326, 0.8141625628409767, 0.77436523339607, 0.7717741998578999, 0.8277794036604611, 0.7633847430102935, 0.7649391602034772, 0.7739074435109674, 0.8551645124332343, 0.9597059695077311, 0.7850914137257743, 1.2977013197277523, 0.9002992968851996, 1.3831444986731338, 0.7331220541572968, 0.7350142393093936, 0.7567455916574858, 0.7385560597014339, 0.7149152120517756, 0.7682996396652781, 0.7393463487075819, 0.7481072036138118, 0.734069907087528, 0.7358011054046295, 0.7224850895555739, 0.7209311980237086, 0.7223641385535655, 0.7444509479751771, 0.7237793899688166, 0.7124175017346824, 0.7100133835816652, 0.7279787393223515, 0.7132861020526218, 0.7215258432218782, 0.7297016004064459, 0.7118851290657459, 0.7130233814154521, 0.7270717290541079, 0.7169446192573943, 0.7159843949491527, 0.7195251762432375, 0.7107730008611063, 0.7205754720071902, 0.7107964111268076, 0.7230118388708694, 0.7108779253052379, 0.7111850826608536, 0.7054410469608705, 0.715691821638958, 0.7116710382128885, 0.7254416655297713, 0.719346414720864, 0.7117348632767134, 0.7095699332483469, 0.7113082096562272, 0.7127274218683907, 0.7003282938304964, 0.7161371056745505, 0.706726851220243, 0.7148098108450531, 0.7072996264626225, 0.7150329235130997, 0.7057746096292351, 0.7074171423290364, 0.7020077083953734, 0.7077634920145933, 0.7237073376941217, 0.7232957204322844, 0.6988248535605363, 0.691888141227348, 0.694514540770047, 0.698407661843785, 0.7052882873367455, 0.703588103037442, 0.7078970833376118, 0.7058342905259203, 0.7085837192665386, 0.7053744224449069]
    print(min(wer_overall))
    plt.figure(0)
    #plt.plot(loss_overall, color='red')
    plt.plot(wer_overall, color='blue')
    plt.title('CNN+RNN ASR Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('WER')
    plt.show()
    plt.figure(1)
    plt.plot(loss_overall, color='red')
    plt.title('CNN+RNN ASR Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

