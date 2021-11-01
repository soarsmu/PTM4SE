import torch.nn as nn
from transformers import BertForSequenceClassification,RobertaConfig,BertConfig


class RoBERTa(nn.Module):
    def __init__(self, bert_model):
        super(RoBERTa, self).__init__()
        # self.encoder = RobertaForSequenceClassification.from_pretrained(bert_model)


        configuration = BertConfig.from_pretrained(bert_model)
        configuration.num_labels = 3
        self.encoder = BertForSequenceClassification(configuration).from_pretrained(bert_model,config=configuration)
    def forward(self, text, label):
        #print(' train label:', label)
        output = self.encoder(text, labels=label)
        #print('train out:', output)
        loss = output.loss
        test_fea = output.logits
        return loss, test_fea
