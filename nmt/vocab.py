"""NMT task reconstruct
produced by: Zhiyu
inspired by Stanford CS224n assignment4

"""

import json
from typing import List, Dict, Tuple
from collections import Counter
from itertools import chain

import torch


class VocabEntry(object):
    """construct source or target vocabulary.
    """
    def __init__(self, word2id: Dict=None):
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = {}
            self.word2id['<pad>'] = 0
            self.word2id['<s>'] = 1
            self.word2id['</s>'] = 2
            self.word2id['<unk>'] = 3
        self.pad_token = '<pad>'
        self.start_token = '<s>'
        self.end_token = '</s>'
        self.pad_id = self.word2id['<pad>']
        self.start_id = self.word2id['<s>']
        self.end_id = self.word2id['</s>']
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}
    
    def __setitem__(self, word):
        """vocabulary can't set new items by indexing keys.
        """
        raise ValueError("Vocabulary is only readable, if you want to set new k-v pair, use vocab.add()")
        
    def __getitem__(self, word: str):
        return self.word2id[word]
    
    def __contains__(self, word:str):
        return word in self.word2id
    
    def __len__(self):
        return len(self.word2id)
    
    def add(self, word: str):
        if word not in self:
            current_id = len(self)
            self.word2id[word] = current_id
            self.id2word[current_id] = word
        else:
            raise ValueError('warning: word [%s] to be added already exists, something might wrong'%word)
            
    def word_from_id(self, idx: int) -> str:
        """get corresponding word from id.
        :return word (str)
        """
        try:
            word = self.id2word[idx]
        except:
            raise KeyError("invalid index to get corresponding word...")
        return word
    
    def get_pad_info(self, index):
        """return represent and index of vocabulary's pad token
        """
        if index == 0:
            return self.pad_token
        elif index == 1:
            return self.pad_id
        else:
            raise ValueError("Wrong index for get pad token information......")
    
    def get_eos_info(self, index):
        if index == 0:
            return self.start_id
        elif index == 1:
            return self.end_id
        else:
            raise ValueError("Wrong index for get information......")
    
    def get_words(self):
        """get all of the words in VocabEntry.
        :return List[str]
        """
        return [self.id2word[idx] for idx in range(len(self))]
            
    @staticmethod
    def build(corpus: List[List[str]], size=5000, freq_cutoff=5):
        """construct vocabulary from corpus.
        :param corpus list[list[str]]: list of sentences of one language
        :param size (int): vocabulary size
        :param freq_cutoff (int): ignore the words whose frequency is less than freq_cutoff
        
        :return vocab (VocabEntry): constructed vocabulary
        """
        vocab = VocabEntry()
        word2freq = Counter(chain(*corpus))
        word2freq = {word: freq for word, freq in word2freq.items() if freq > freq_cutoff}
        words_selected = sorted(word2freq.keys(), key=lambda w: word2freq[w], reverse=True)[:size]
        for w in words_selected:
            vocab.add(w)
        print("vocabulary constructing completed, %d/%d words included......" % (len(words_selected), len(word2freq)))
        return vocab        
    
    def pad_sents(self, sents: List[List[int]], resource: str) -> List[List[int]]:
        """pad batch of sentences to its maximum length.
        :param sents (List[List[int]]): sentences to be padded
        :param resource (str): 'src' or 'tgt', source which sentences come from
        
        :return sents_padded (List[List[int]]): sentences padded, and is not arranged by decreasing length.
        :return sents_len (List[int]): length of each sentence, including <s>/</s> if exist
        """
        assert resource in ['src', 'tgt'], "wrong resource choice, only 'src' or 'tgt'"
        
        max_length = max(len(s) for s in sents)
        if resource == 'tgt': max_length += 2
        # sents = sorted(sents, key=lambda s: len(s), reverse=True)
        sents_padded = []
        sents_len = []
        for s in sents:
            if resource == 'tgt':
                s = [self.word2id['<s>']] + s + [self.word2id['</s>']]
            sents_len.append(len(s))
            s_padded = s[:] + [self.pad_id]*(max_length-len(s))
            sents_padded.append(s_padded)
        return sents_padded, sents_len
    
    def to_tensor(self, sents: List[List[str]], resource: str) -> torch.tensor:
        """transform sentences to tensor.
        :param sents (List[List[str]]): original sentences without padding
        :param resource (str): sentence come from, 'src' or 'tgt'
        
        :return sents_tensor (batch, max_sent_length): the sentences' each element is word's corresponding index
        :return sents_len (torch.tensor): length of each sentence
        """
        sents_id = []
        for s in sents:
            s_id = [self.word2id.get(word, self.unk_id) for word in s]
            sents_id.append(s_id)
        sents_id, sents_len = self.pad_sents(sents_id, resource)
        sents_tensor = torch.tensor(sents_id, dtype=torch.long)
        sents_len = torch.tensor(sents_len, dtype=torch.int)
        return sents_tensor, sents_len
    
    def to_sentences(self, sents_id: List[List[int]]) -> List[List[str]]:
        """transform id back to strings.
        """
        sents = []
        for s_id in sents_id:
            s = []
            for w_id in s_id:
                s.append(self.word_from_id(w_id))
            sents.append(s)
        return sents
    
    
class Vocab(object):
    """absorb both src and tgt VocabEntry into a single class.
    """
    def __init__(self, src: VocabEntry, tgt: VocabEntry):
        self.src = src
        self.tgt = tgt
        
    def __repr__(self):
        """Vocab entity representation.
        """
        return "vocabulary: source -- %d words; target -- %d words" % (len(self.src), len(self.tgt))
    
    def save(self, fpath='./vocab.json'):
        json.dump(dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id), open(fpath, 'w'), indent=2)
        
    @staticmethod
    def load(fpath='./vocab.json'):
        entry = json.load(open(fpath, 'r'))
        src_word2id = entry['src_word2id']
        tgt_word2id = entry['tgt_word2id']
        return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))
    
if __name__ == '__main__':
    print("sanity check for VocabEntry class......")
    sentences = ["I have a dream".split(), "I need to be faster".split(), "I need to be strong and healthy".split()]
    print("source sentence:")
    print(sentences)
    print("generate vocab 0 for cutoff 1...")
    vocab = VocabEntry.build(sentences, size=100, freq_cutoff=1)
    print(vocab.get_words())
    print("generate vocab 1 for cutoff 0...")
    vocab = VocabEntry.build(sentences, size=100, freq_cutoff=0)
    print(vocab.get_words())
    sents_tensor, sents_len = vocab.to_tensor(sentences, 'tgt')
    print("padded sents_tensor:\n", sents_tensor, "\nsize:", sents_tensor.size())
    print("true length:", sents_len)
    print("transform back to original sentence......")
    print(vocab.to_sentences(sents_tensor.numpy())) 