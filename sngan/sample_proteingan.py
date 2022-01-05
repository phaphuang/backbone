#    Copyright (C) 2018 Anvita Gupta
#
#    This program is free software: you can redistribute it and/or  modify
#    it under the terms of the GNU Affero General Public License, version 3,
#    as published by the Free Software Foundation.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

from sklearn.preprocessing import OneHotEncoder
import os, math, glob, argparse
from utils.torch_utils import *
from utils.utils import *
import matplotlib.pyplot as plt
import utils.language_helpers
plt.switch_backend('agg')
import numpy as np
from models.discriminator import ResnetDiscriminator
from models.generator import ResnetGenerator

from utils.blast_summary import get_protein_sequences, sequences_to_fasta, get_local_blast_results, update_sequences_with_blast_results, get_stats

#### Higne Loss src: https://github.com/heykeetae/Self-Attention-GAN/blob/8714a54ba5027d680190791ba3a6bb08f9c9a129/trainer.py
#### Non saturate loss src: https://github.com/SiskonEmilia/StyleGAN-PyTorch/blob/master/train.py

blast_path = "/content/drive/MyDrive/ECUST/Projects/ProteinGAN/results/mdh/proteingan"
f_log_name = "proteingan_blomsum45.log"
#f_loss_name = "protein_gan_loss.log"

class SNGAN_Bio():
    def __init__(self, batch_size=64, lr=0.0001, num_epochs=2000, seq_len = 512, data_dir='../data/bmdh_seq_uniprot_single_class.fasta', \
        run_name='test', hidden=512, d_steps = 1, g_steps = 1):
        self.hidden = hidden
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = num_epochs
        self.seq_len = seq_len
        self.d_steps = d_steps
        self.g_steps = g_steps
        #self.checkpoint_dir = './checkpoint/' + run_name + "/"
        self.checkpoint_dir = blast_path + "/checkpoint/"
        #self.sample_dir = './samples/' + run_name + "/"
        self.sample_dir = blast_path + '/samples/'
        self.load_data(data_dir)
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir): os.makedirs(self.sample_dir)
        self.use_cuda = True if torch.cuda.is_available() else False
        self.build_model()

    def build_model(self):
        self.G = ResnetGenerator()
        self.D = ResnetDiscriminator()
        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()
        print(self.G)
        print(self.D)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.0, 0.9))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.0, 0.9))

    def load_data(self, datadir):
        max_examples = 1e6
        lines, self.charmap, self.inv_charmap = utils.language_helpers.load_dataset(
            max_length=self.seq_len,
            max_n_examples=max_examples,
            data_dir=datadir
        )
        self.data = lines

    def save_model(self, epoch):
        torch.save(self.G.state_dict(), self.checkpoint_dir + "G_weights_{}.pth".format(epoch))
        torch.save(self.D.state_dict(), self.checkpoint_dir + "D_weights_{}.pth".format(epoch))

    def load_model(self, directory = ''):
        '''
            Load model parameters from most recent epoch
        '''
        if len(directory) == 0:
            directory = self.checkpoint_dir
        list_G = glob.glob(directory + "G*.pth")
        list_D = glob.glob(directory + "D*.pth")
        if len(list_G) == 0:
            print("[*] Checkpoint not found! Starting from scratch.")
            return 1 #file is not there
        G_file = max(list_G, key=os.path.getctime)
        D_file = max(list_D, key=os.path.getctime)
        epoch_found = int( (G_file.split('_')[-1]).split('.')[0])
        print("[*] Checkpoint {} found at {}!".format(epoch_found, directory))
        self.G.load_state_dict(torch.load(G_file))
        self.D.load_state_dict(torch.load(D_file))
        return epoch_found

    def sample(self):
        z = to_var(torch.randn(self.batch_size, 128))
        self.G.eval()
        torch_seqs = self.G(z)
        torch_seqs = torch.squeeze(torch_seqs.permute((0, 3, 2, 1)))
        #torch_seqs = torch.squeeze(torch.argmax(torch_seqs, dim=-1))
        seqs = (torch_seqs.data).cpu().numpy()
        decoded_seqs = [decode_one_seq(seq, self.inv_charmap)+"\n" for seq in seqs]
        with open(self.sample_dir + "sampled_{}.txt".format(step), 'w+') as f:
            f.writelines(decoded_seqs)

def main():
    parser = argparse.ArgumentParser(description='ResNetProteinGAN for producing gene sequences.')
    parser.add_argument("--run_name", default= "bacteria_mdh_gen", help="Name for output files (checkpoint and sample dir)")
    parser.add_argument("--load_dir", default="", help="Option to load checkpoint from other model (Defaults to run name)")
    args = parser.parse_args()
    model = SNGAN_Bio(run_name=args.run_name)
    model.train_model(args.load_dir)

if __name__ == '__main__':
    main()