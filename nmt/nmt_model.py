"""NMT task reconstruct
produced by: Zhiyu
inspired by Stanford CS224n assignment4

"""

from typing import List, Dict, Tuple
from collections import namedtuple
from copy import deepcopy

import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from vocab import VocabEntry, Vocab

Pair = namedtuple('Pair', 'sent score last_state last_out')  # stores a sentence(each element is vocab index), 
                                         # its predicting likelihood, its last hidden/cell state and last output

def load_word2vec(fpath: str, vocab: VocabEntry, device: torch.device) -> torch.tensor:
    """load pretrained embedding vectors for words in vocab.
    :param fpath : word2vec file(from fasttext) path, in which already contains </s> token.
    :param vocab : constructed vocabulary
    :return word2vec (vocab_size, embed_size): tensor of word2vec
    """
    print("loading pretrained word2vec from %s......"%fpath)
    model = KeyedVectors.load_word2vec_format(fpath, limit=int(1e5))
    words = vocab.get_words()
    word2vec = []
    for w in tqdm(words, desc='loading'):
        try:
            word2vec.append(model[w].astype(np.float))
        except KeyError:
            if w == vocab.get_pad_info(0):
                # initialize pad token with zero vector
                word2vec.append(np.zeros(model.vector_size, dtype=np.float))
            else:
                uniform_init = 0.1 
                word2vec.append(np.random.uniform(low=-uniform_init, high=uniform_init, size=model.vector_size).astype(np.float))
    word2vec = np.stack(word2vec, axis=0)
    word2vec = torch.from_numpy(word2vec).to(torch.float).to(device)
    assert word2vec.size(0) == len(vocab), "tensor size wrong, first dimention should be equal to vocab size"
    return word2vec


class Encoder(nn.Module):
    """Encoder module to get source sentence representations.
    in this module, bidirectional-LSTM layer is used.
    """
    def __init__(self, input_size, hidden_size, atten_size):
        """note,
        :param hidden_size (int): encoder and decoder's hidden_size should be equal
        :param atten_size (int): the projection dimention for attention computation. encoder and decoder's atten_size must match
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.encode = nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=True)
        self.proj_hidden = nn.Linear(2*hidden_size, hidden_size, bias=False)  # project encoder's last_hidden_state to decoder's init_hidden_state
        self.proj_cell = nn.Linear(2*hidden_size, hidden_size, bias=False)  # project encoder's last_cell_state to decoder's init_cell_state
        self.w_atten = nn.Parameter(torch.empty(hidden_size*2, atten_size))  # compute matrix multiplication for encoder output ahead in later attention computation
        
    def forward(self, source: torch.tensor, src_len: torch.tensor, 
                init_encode_state: Tuple[torch.tensor, torch.tensor]=None):
        """do source encoding and produce corresponding representations.
        :param source (max_sent_len, batch, embed_size): source sentences to be translated, length with decreasing order
        :param src_len : each sentence actual length.
        :param init_encode_state tuple(tensor, tensor): first one is initial hidden state, and the second one is initial cell state.
            both of them, the size is (num_layers*num_directions, batch, hidden_size).
        
        :return memory (max_sent_len, batch, num_directions*hidden_size): representations for each time step
        :return init_decoder_state tuple(tensor, tensor): decoder's initial hidden_state(first one) and cell_state(second one).
            both of them, the size is (1, batch, hidden_size)
        :return atten_vec (batch, max_sent_len, atten_size): tensor used for later attention computation
        """
        # print("encoder get data size: {}, {}".format(source.size(), src_len.size()))
        
        source = pack_padded_sequence(source, src_len)
        if init_encode_state:
            self.encode.flatten_parameters()
            memory, last_state = self.encode(source, init_encode_state)
        else:
            self.encode.flatten_parameters()
            memory, last_state = self.encode(source)
        memory, _ = pad_packed_sequence(memory)
        last_hidden_state = torch.transpose(last_state[0], 0, 1).contiguous().view(-1, 2*self.hidden_size)
        last_cell_state = torch.transpose(last_state[1], 0, 1).contiguous().view(-1, 2*self.hidden_size)
        dec_hidden = self.proj_hidden(last_hidden_state).unsqueeze(0)
        dec_cell = self.proj_cell(last_cell_state).unsqueeze(0)
        init_decoder_state = (dec_hidden, dec_cell)
        atten_vec = torch.matmul(torch.transpose(memory, 0, 1), self.w_atten)  # (batch, max_sent_len, atten_size)
        
        return memory, init_decoder_state, atten_vec
        
    
class Decoder(nn.Module):
    """Generate target sentences from source sentences.
    In each step, the actual input for RNN network is the word embedding combined with last step's output.
    """
    def __init__(self, input_size, hidden_size, atten_size, dropout_rate=0.5):
        """note, hidden_size and atten_size should be equal to encoder's.
        :param input_size (int): word's embed_size + last step's output dimention
        """
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.decode = nn.LSTM(input_size, hidden_size, num_layers=1)
        self.w_atten = nn.Parameter(torch.empty(hidden_size, atten_size)) 
        self.combined_proj = nn.Linear(3*hidden_size, hidden_size)
    
    @property
    def device(self) -> torch.device:
        return self.w_atten.device
    
    def forward(self, target: torch.tensor, dec_last_state: Tuple[torch.tensor, torch.tensor],
                memory: torch.tensor, enc_atten_vec: torch.tensor, src_len: torch.tensor):
        """Single step translate, generate present step's combined output based on previous outputs and present word embedding.
        :param target (1, batch, input_size): input present step, the input is a concatenation result, always embed vector first!!!
        :param dec_last_state tuple(tensor, tensor): last hidden and cell state, each size (num_layers*1, batch, hidden_size)
        :param memory (src_max_sent_len, batch, 2*hidden_size): source sentences representation
        :param enc_atten_vec (batch, src_max_sent_len, atten_size)
        :param src_len : actual length of source sentences.
        
        :return output (batch, hidden_size)
        :return dec_state tuple(tensor, tensor): hidden_sate and cell_state, each is (num_layers*1, batch, hidden_size)
        """
        # print("decoder get size: {}, {}, {}, {}".format(target.size(), memory.size(), enc_atten_vec.size(), src_len.size()))
        assert target.size(1) == memory.size(1) == enc_atten_vec.size(0) == len(src_len), "param shape doesn't match with requirements"
        mask = torch.zeros(target.size(1), memory.size(0), device=self.device)
        for idx, L in enumerate(src_len):
            mask[idx, L.item():] = float('-inf')
        mask = mask.unsqueeze(1)
        self.decode.flatten_parameters()
        x, dec_state = self.decode(target, dec_last_state)
        atten_result = self.attention(x, memory, enc_atten_vec, mask)
        combined_vec = torch.cat((x.squeeze(0), atten_result.squeeze(1)), dim=1)  # (batch, 3*hidden_size)
        output = F.dropout(torch.tanh(self.combined_proj(combined_vec)), p=self.dropout_rate)
        
        return output, dec_state
        
        
    def attention(self, dec_out: torch.tensor, memory: torch.tensor, 
                  enc_atten_vec: torch.tensor, memory_mask: torch.tensor):
        """attention computation.
        :param dec_out (1, batch, hidden_size): output of recurrent module of decoder
        :param memory (src_max_sent_len, batch, 2*hidden_size): source sentences representation
        :param enc_atten_vec (batch, src_max_sent_len, atten_size)
        :param memory_mask (batch, 1, src_max_sent_len): with '-inf' corresponding to source <pad> token, others are 0.
        
        :return result (batch, 1, 2*hidden_size): attention result
        """
        assert dec_out.size(1) == memory.size(1), "source and target batch size doesn't match"
        assert enc_atten_vec.size(-1) == self.w_atten.size(-1), "encoder and decoder attention size doesn't match"
        dec_atten_vec = torch.matmul(torch.transpose(dec_out, 0, 1), self.w_atten)  # (batch, 1, atten_size)
        atten_score = torch.matmul(dec_atten_vec, torch.transpose(enc_atten_vec, 1, 2)) # (batch, 1, src_max_sent_len)
        atten_score = atten_score+memory_mask  # mask out <pad> elements
        atten_distribut = F.softmax(atten_score, dim=-1)  # (batch, 1, src_max_len)
        result = torch.matmul(atten_distribut, torch.transpose(memory, 0, 1)) 
        
        return result
        
        
class NMT(nn.Module):
    """neural machine translation model.
    """
    def __init__(self, hidden_size, atten_size, vocab: Vocab, dropout_rate=0.5, word2vec=None, embed_size=300):
        """
        :param hidden_size (int): for both encoder and decoder.
        :param atten_size (int): intermediat dimention when doing attention
        :param vocab : entity contain both source and target vocabularies.
        :param word2vec tuple(tensor, tensor): pretrained word2vec models, [0] for source, [1] for target. the dim0 is 
            corresponding to vocab length. 
        :param embed_size (int): this parameter is effective only when word2vec is None.
        """
        super(NMT, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.atten_size = atten_size
        self.dropout_rate = dropout_rate
        if word2vec:
            self.word2vec_pretrained = True
            self.embed_size = word2vec[0].size(1)
            self.src_embedding = nn.Embedding.from_pretrained(word2vec[0], freeze=False, padding_idx=vocab.src.get_pad_info(1))
            self.tgt_embedding = nn.Embedding.from_pretrained(word2vec[1], freeze=False, padding_idx=vocab.tgt.get_pad_info(1))
        else:
            self.word2vec_pretrained = False
            self.embed_size = embed_size
            self.src_embedding = nn.Embedding(len(vocab.src), embed_size, padding_idx=vocab.src.get_pad_info(1))
            self.tgt_embedding = nn.Embedding(len(vocab.tgt), embed_size, padding_idx=vocab.tgt.get_pad_info(1))
        self.encoder = Encoder(self.src_embedding.weight.size(1), hidden_size, atten_size)
        self.decoder = Decoder(self.tgt_embedding.weight.size(1)+hidden_size, hidden_size, atten_size, dropout_rate)
        self.proj_vocab = nn.Linear(hidden_size, len(vocab.tgt))
     
    @property
    def device(self) -> torch.device:
        return self.src_embedding.weight.device
        
    def forward(self, source: torch.tensor, src_len: torch.tensor, 
                target: torch.tensor, tgt_len: torch.tensor):
        """complete source to target training processing, based on encoder-decoder frame
        :param source (batch, src_max_len): source sentences with lengths in decreasing order
        :param src_len : tensor, the actual length for source sentences
        :param target (batch, tgt_max_len): target translated sentences corresponding to source. 
        :param tgt_len : tensor, the actual length for target sentences, the count including '</s>'.
        
        :return loss (scalar tensor): total predicting loss
        """
        # print("model get data size: {}, {}, {}, {}".format(source.size(), src_len.size(), target.size(), tgt_len.size()))
        source = source.t()
        target = target.t()
        
        tgt_max_len, batch_size = target.size()
        src_embedded = self.src_embedding(source)
        memory, init_dec_state, enc_atten_vec = self.encoder(src_embedded, src_len)
        predict = []
        t = 0
        for batch_words in target[:-1]:
            words_embedded = self.tgt_embedding(batch_words)  # (batch, embed_size)
            if t == 0:
                last_out = torch.zeros(batch_size, self.hidden_size, dtype=torch.float, device=self.device)
            x_in = torch.cat((words_embedded, last_out), dim=1).unsqueeze(0)
            # note here the init_dec_state is transfered only for 1 layer rnn case, otherwise extra process needed
            output, dec_state = self.decoder(x_in, init_dec_state, memory, enc_atten_vec, src_len)
            p = F.log_softmax(self.proj_vocab(output), dim=-1, dtype=torch.float)  # (1, batch, tgt_vocab_size)
            predict.append(p)
            
            last_out = output
            init_dec_state = dec_state
            t += 1
        predict = torch.stack(predict, dim=0)  # (tgt_max_len-1, batch, tgt_vocab_size)
        
        # computing prediction total loss
        tgt_mask = torch.tensor([[1]*l.item()+[0]*(tgt_max_len-1-l.item()) for l in (tgt_len-1)], dtype=torch.int, device=self.device)
        gold_standard = target[1:].t().unsqueeze(-1)  # (batch, tgt_max_len-1, 1)
        label_prob = torch.gather(predict.transpose(0, 1), index=gold_standard, dim=-1).squeeze(-1) * tgt_mask
        # loss = -label_prob.sum()
        loss = -label_prob
        return loss
    
    def translate(self, source: torch.tensor, beam_size=5, max_len=70):
        """single source-target sentence translate.
        :param source (1, src_len): one source sentence to be translated.
        :param beam_size : beam search size
        
        :return translate_result (list[int]): most likely candidate
        """
        source = source.t()
        src_len = torch.tensor([source.size(0)], dtype=torch.int, device=self.device)
        src_embedded = self.src_embedding(source)
        memory, init_dec_state, enc_atten_vec = self.encoder(src_embedded, src_len)
        
        # ----beam search----
        # search stop when number of completed sentences reaches beam_size, or get to max_len.
        completed = []
        last_out = torch.zeros(self.hidden_size, dtype=torch.float, device=self.device)
        candidate = [Pair([self.vocab.tgt.get_eos_info(0)], 0, init_dec_state, last_out)]
        step = 0
        while len(completed)<beam_size and step<max_len:
            step += 1
            # generate batch input from candidate sentences
            last_outs = torch.stack([item.last_out for item in candidate], dim=0)  # (num_candidate, hidden_size)
            words = torch.tensor([item.sent[-1] for item in candidate], dtype=torch.long, device=self.device)
            words_embed = self.tgt_embedding(words)  # (num_candidate, embed_size)
            x_in = torch.cat((words_embed, last_outs), dim=1).unsqueeze(0)  # (1, num_candidate, embed_size)
            last_hidden_state = torch.cat([item.last_state[0] for item in candidate], dim=1)  # (num_layers, num_candidate, hidden_size)
            last_cell_state = torch.cat([item.last_state[1] for item in candidate], dim=1)  # (num_layers, num_candidate, hidden_size)
            
            # expand memory, enc_atten_vec and src_len to match the batch size.
            batch_size = len(candidate)
            memory_expand = memory.expand(-1, batch_size, -1)
            enc_atten_vec_expand = enc_atten_vec.expand(batch_size, -1, -1)
            src_len_expand = src_len.expand(batch_size)
            
            # predict the next words for all of the candidates
            output, dec_state = self.decoder(x_in, (last_hidden_state, last_cell_state), memory_expand, enc_atten_vec_expand, src_len_expand)
            prob = F.log_softmax(self.proj_vocab(output), dim=-1)  # (num_candidate, vocab_size) 
            scores = prob + torch.tensor([item.score for item in candidate], dtype=torch.float, device=self.device).view(-1, 1)
            
            # select the top 'beam_size' predicts and update candidates
            tops = torch.topk(scores.view(-1), beam_size)
            candidate_new = []
            for score, idx in zip(*tops):
                cand_id = idx.item() // len(self.vocab.tgt)
                vocab_id = idx.item() % len(self.vocab.tgt)
                if vocab_id == self.vocab.tgt.get_eos_info(1):
                    # the completed sentence doesn't include </s> token
                    completed.append(Pair(candidate[cand_id].sent, score, None, None))
                else:
                    last_state = (dec_state[0][:, cand_id, :].unsqueeze(1), dec_state[1][:, cand_id, :].unsqueeze(1))
                    candidate_new.append(Pair(candidate[cand_id].sent+[vocab_id], score, last_state, output[cand_id]))
            candidate = candidate_new
        
        if len(completed) < beam_size:
            completed.extend(candidate[:beam_size-len(completed)])
        ave_scores = [item.score/(len(item.sent)-1) for item in completed]
        translate_result = completed[np.argmax(ave_scores)].sent[1:]
        
        return translate_result
                
    def init(self):
        """initialize model parameters. """
        print("initializing model parameters......")
        for sub_m in self.children():
            if self.word2vec_pretrained and type(sub_m)==nn.Embedding:
                print("skip Embedding layer initialization, for pretrained word2vec is used.")
                continue
            for param in sub_m.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, a=0, b=0.1)
                   
    def save(self, fpath):
        print("save model parameters to %s"%fpath)
        params = {'args': dict(hidden_size=self.hidden_size, atten_size=self.atten_size, dropout_rate=self.dropout_rate, embed_size=self.embed_size),
               'state_dict': self.state_dict()}
        torch.save(params, fpath+'/model.bin')
    
    @staticmethod
    def load(fpath, vocab: Vocab):
        """load model from file."""
        params = torch.load(fpath+'/model.bin', map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=vocab, **args)
        model.load_state_dict(params['state_dict'])
        return model