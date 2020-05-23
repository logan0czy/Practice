"""NMT task reconstruct
produced by: Zhiyu
inspired by Stanford CS224n assignment4

Usage:
    run.py train [options]
    run.py test [options]
    
Options:
    -h --help                     show this help document
    --data_path=<fpath>           the folder where corpus is stored [default: ./en_es_data]
    --model_path=<fpath>          the folder where model is saved [default: /home/ubuntu/MyFiles/myProj/nmt]
    --use_word2vec                whether use pretrained word2vec
    --word2vec_fpath=<fpath>      folder of pretrained word2vec [default: /home/ubuntu/MyFiles/myData]
    --loss_info_interval=<int>    interval to print average training loss information [default: 20]
    --dev_info_interval=<int>     interval to validate on development data [default: 1000]
    --max_epoch=<int>             maximum training epoches [default: 10]
    --corpus_limit=<int>          lines to load from corpus [default: None] 
    --vocab_size=<int>            size of vocabulary for both source and target [default: 5000]
    --freq_cutoff=<int>           ignore words whose freq under this value [default: 5] 
    --batch_size=<int>            training samples batch size [default: 32] 
    --hidden_size=<int>           hidden layer size in RNN models [default: 512] 
    --atten_size=<int>            intermediate attention computation dimention [default: 256] 
    --dropout_rate=<float>         layer dropout rate [default: 0.5]
    --embed_size=<int>            words embedding size [default: 300] 
    --lr=<float>                  learning rate [default: 0.001] 
    --lr_decay=<float>            learning rate decay [default: 0.9] 
    --grad_clip=<float>           maximun gradient norm [default: 5.0] 
    --patience=<int>              time to wait when new result is not better than previous ones [default: 5]
    --trial=<int>                 time to try restart from previous best model [default: 5]
    --beam_size=<int>              beam search size [default: 5]
    --max_len=<int>               maximum translate sentence length [default: 70]
    --num_workers=<int>            processes to load data [default: 0]
"""

import math
import time

from tqdm import tqdm
from docopt import docopt
from prefetch_generator import BackgroundGenerator
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import nltk
nltk.data.path.append('/home/ubuntu/MyFiles/myData/nltk_data')
from mosestokenizer import *

from vocab import VocabEntry, Vocab
from nmt_model import load_word2vec, NMT


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super(DataLoaderX, self).__iter__())
    

class WrappedDataLoader(object):
    """modified dataloader, in which the batch sentences is sorted by length.
    """
    def __init__(self, dataloader):
        self.DL = dataloader
        
    def __len__(self):
        return len(self.DL)
    
    def __iter__(self):
        """generate batch data."""
        data = iter(self.DL)
        for batch in data:
            yield self.sort(batch)
    
    def sort(self, batch):
        """sort the sentences in batch by their real length.
        :param batch (tuple): have (source, src_len, target, tgt_len), four tensors
        :return  (tuple): rearranged tensors corresponding to sorted result of src_len.
        """
        source, src_len, target, tgt_len = batch
        _, indices = torch.sort(src_len, descending=True)
        return (source[indices], src_len[indices], target[indices], tgt_len[indices])
    
    
def load_corpus(fpath, language, limit=None):
    """load sentences from file and tokenize them.
    :param fpath (str): the folder where the corpus stores.
    :param language (str): what kind of language.
    :param limit (int): # of lines to read
    
    :return corpus (list[list[str]]): all of the tokenized sentences.
    """
    corpus = []
    with open(fpath, 'r') as f:
        num = 0
        tokenizer = MosesTokenizer(language)
        for line in f:
            corpus.append(tokenizer(line))
            num += 1
            if limit and num>=limit: break
        tokenizer.close()
    return corpus


def generate_dl(data_path, data_class, vocab, batch_size, num_workers=0, limit=None):
    """load training or development data and generate dataloader.
    :param data_path (str): folder where the data saved.
    :param data_class (str): 'train' or 'dev'
    :param vocab (Vocab): stores source and target vocabulary.
    :param batch_size (int)
    :param num_workers (int): num of subprocess used for data loading.
    
    :return dataloader (WrappedDataLoader or DataLoader): contains source sentences and target sentences.
    """
    assert data_class == 'train' or data_class == 'dev', "wrong choice for data class to be processed"
    if data_class == 'train':
        src_corpus = load_corpus(data_path+'/train.es', 'es', limit)
        tgt_corpus = load_corpus(data_path+'/train.en', 'en', limit)
    else:
        src_corpus = load_corpus(data_path+'/dev.es', 'es', limit)
        tgt_corpus = load_corpus(data_path+'/dev.en', 'en', limit)
        
    source, src_len = vocab.src.to_tensor(src_corpus, 'src')
    target, tgt_len = vocab.tgt.to_tensor(tgt_corpus, 'tgt')
    dataset = TensorDataset(source, src_len, target, tgt_len)
    
    if data_class == 'train':
        dataloader = DataLoaderX(dataset, batch_size, shuffle=True, 
                        num_workers=num_workers, pin_memory=False)
        dataloader = WrappedDataLoader(dataloader)
    else:
        dataloader = DataLoaderX(dataset, batch_size*2, shuffle=False, 
                        num_workers=num_workers, pin_memory=False)
        dataloader = WrappedDataLoader(dataloader)
    return dataloader
    

def train(args):
    data_path = args['--data_path']
    model_path = args['--model_path']
    use_word2vec = args['--use_word2vec']
    word2vec_fpath = args['--word2vec_fpath']
    loss_info_interval = int(args['--loss_info_interval'])
    dev_info_interval = int(args['--dev_info_interval'])
    max_epoch = int(args['--max_epoch'])
    num_workers = int(args['--num_workers'])
    
    corpus_limit = None if args['--corpus_limit']=='None' else int(args['--corpus_limit'])
    vocab_size = int(args['--vocab_size'])
    freq_cutoff = int(args['--freq_cutoff'])
    batch_size = int(args['--batch_size'])
    hidden_size = int(args['--hidden_size'])
    atten_size = int(args['--atten_size'])
    dropout_rate = float(args['--dropout_rate'])
    embed_size = int(args['--embed_size'])
    lr = float(args['--lr'])
    lr_decay = float(args['--lr_decay'])
    grad_clip = float(args['--grad_clip'])
    patience = int(args['--patience'])
    trial = int(args['--trial'])
    
    # create vocabulary
    print("create source and target vocabulary...")
    es_corpus = load_corpus(data_path+'/train.es', 'es', corpus_limit)
    vocab_src = VocabEntry.build(es_corpus, vocab_size, freq_cutoff)
    en_corpus = load_corpus(data_path+'/train.en', 'en', corpus_limit)
    vocab_tgt = VocabEntry.build(en_corpus, vocab_size, freq_cutoff)
    vocabulary = Vocab(vocab_src, vocab_tgt)
    vocabulary.save()
#     vocabulary = Vocab.load()
    
    # create dataloader
    print("create dataloader...")
    train_dl = generate_dl(data_path, 'train', vocabulary, batch_size, num_workers, corpus_limit)
    dev_dl = generate_dl(data_path, 'dev', vocabulary, batch_size, num_workers, corpus_limit)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # load pretrained word2vec
    print("load pretrained word2vec...")
    if use_word2vec:
        word2vec_src = load_word2vec(word2vec_fpath+'/cc.es.300.vec', vocabulary.src, device)
        word2vec_tgt = load_word2vec(word2vec_fpath+'/cc.en.300.vec', vocabulary.tgt, device)
        word2vec = (word2vec_src, word2vec_tgt)
        print("size: src -- {} tgt -- {}".format(word2vec_src.size(), word2vec_tgt.size()))
    else:
        word2vec = None
        
    # create model and optimizer
    model = NMT(hidden_size, atten_size, vocabulary, dropout_rate, word2vec, embed_size)
    model.init()
    
#     model = NMT.load(model_path, vocabulary)
    
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("- - - - - - - - >%d GPUs are being used now!!!"%torch.cuda.device_count())
        model_parallel = nn.DataParallel(model)
    opt = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.99))
    
#     opt.load_state_dict(torch.load(model_path+'/model.bin.optim', map_location=lambda storage, loc: storage))
    
    print("now we start to train this model ...... batch_size %d"%batch_size)
    start_time = train_time = time.time()
    epoch = iter_cum = example_cum = loss_cum = words_cum = sents_cum = 0
    n_patience = n_trial = 0
    hist_best_score = float('inf')
    while True:
        epoch += 1
        if epoch>max_epoch: 
            print("reached maximum number of epoches......")
            exit(0)
        for source, src_len, target, tgt_len in train_dl:
            model_parallel.train()
            # print("input size to model {}, {}, {}, {}".format(source.size(), src_len.size(), target.size(), tgt_len.size()))
            
            loss = model_parallel(source.to(device), src_len.to(device), target.to(device), tgt_len.to(device))
            # print("output loss size: {}".format(loss.size()))
            loss = loss.sum()
            loss_avg = loss / len(tgt_len)
            loss_avg.backward()
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            opt.step()
            opt.zero_grad()
            iter_cum += 1
            example_cum += len(tgt_len)
            loss_cum += loss.item()
            words_cum += (tgt_len-1).sum().item()
            sents_cum += len(tgt_len)
            
            if iter_cum % loss_info_interval == 0:
                pre_time = time.time()
                print("epoch: %d, iter: %d, cum. example: %d, avg. loss: %.2f, avg. ppl: %.2f, speed: %.2fwords/sec, time_eclapsed: %d sec"%
                      (epoch, iter_cum, example_cum, loss_cum/sents_cum, math.exp(loss_cum/words_cum), words_cum/(pre_time-train_time), pre_time-start_time))
                train_time = time.time()
                loss_cum = words_cum = sents_cum = 0
        
            if iter_cum % dev_info_interval == 0:
                print("validation begin ......")
                model_parallel.eval()
                with torch.no_grad():
                    loss_dev = words_dev = sents_dev = 0
                    for source, src_len, target, tgt_len in dev_dl:
                        loss = model_parallel(source.to(device), src_len.to(device), target.to(device), tgt_len.to(device))
                        loss = loss.sum()
                        loss_dev += loss.item()
                        words_dev += (tgt_len-1).sum().item()
                        sents_dev += len(tgt_len)
                    print("avg. loss: %.2f,  avg. ppl: %.2f"%(loss_dev/sents_dev, math.exp(loss_dev/words_dev)))
                
                # compare performance with history
                is_better = hist_best_score > (loss_dev/sents_dev)
                if is_better:
                    print("model improved, saved to %s ......"%model_path)
                    n_patience = 0
                    hist_best_score = loss_dev/sents_dev
                    model.save(model_path)
                    torch.save(opt.state_dict(), model_path+'/model.bin.optim')
                else:
                    n_patience += 1
                    print("hit # %d patience" % n_patience)
                    print("decay learning rate ......")
                    lr = opt.param_groups[0]['lr'] * lr_decay
                    if n_patience > patience:
                        n_trial += 1
                        print("hit # %d trial" % n_trial)
                        if n_trial > trial: 
                            print("early stop!")
                            exit(0)
                        n_patience = 0
                        print("load previous best model")
                        params = torch.load(model_path+'/model.bin', map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model.to(device)
                        if torch.cuda.device_count() > 1:
                            model_parallel = nn.DataParallel(model)
                        opt.load_state_dict(torch.load(model_path+'/model.bin.optim', map_location=lambda storage, loc: storage))
                    for param_group in opt.param_groups:
                        param_group['lr'] = lr
                        

def test(args):
    data_path = args['--data_path']
    model_path = args['--model_path']
    beam_size = int(args['--beam_size'])
    max_len = int(args['--max_len'])
    
    vocab = Vocab.load()
    source = load_corpus(data_path+'/test.es', 'es', limit=100)
    reference_tgt = load_corpus(data_path+'/test.en', 'en', limit=100)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = NMT.load(model_path, vocab)
    model.to(device)
    model.eval()
    
    translate_tgt = []
    with torch.no_grad():
        for src in tqdm(source, desc='translate '):
            src, _ = vocab.src.to_tensor([src], 'src')
            tgt = model.translate(src.to(device), beam_size, max_len)
            translate_tgt.append(tgt)
            
    translate_tgt = vocab.tgt.to_sentences(translate_tgt)
    if reference_tgt[0][0] == '<s>':
        regerence_tgt = [sent[1:-1] for sent in reference_tgt]
    bleu_score = nltk.translate.bleu_score.corpus_bleu([[refer] for refer in reference_tgt], translate_tgt)
    print("corpus bleu score on test data is %.2f" % (bleu_score*100))
    
    # write translate sentences to file
    with open(data_path+'/result.txt', 'w') as f:
        detokenizer = MosesDetokenizer('en')
        for sent in translate_tgt:
            sent = detokenizer(sent)
            f.write(sent+'\n')
        detokenizer.close()
            
if __name__ == '__main__':
    args = docopt(__doc__)
    if args['train']:
        train(args)
    elif args['test']:
        test(args)
    else:
        raise RuntimeError("Invalid mode choice")