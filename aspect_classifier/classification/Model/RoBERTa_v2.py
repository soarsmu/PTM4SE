import torch.nn as nn
from transformers import RobertaForSequenceClassification,RobertaModel,XLNetForSequenceClassification
from transformers import BertForSequenceClassification,RobertaConfig,BertConfig,BertModel

class RoBERTa_V2(nn.Module):
    def __init__(self, bert_model):
        super(RoBERTa_V2, self).__init__()
        # self.bert = RobertaModel.from_pretrained(bert_model)
        self.bert = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels = 2)
        for param in self.bert.parameters():
            param.requires_grad = True
        # drop_out    
        # self.dropout = nn.Dropout(0.15)
        # self.fc = nn.Linear(768, 2)

        
    def forward(self, text):
        _, pooled = self.bert(text, return_dict=False)
        pooled = self.dropout(pooled)
        out = self.fc(pooled)
        return out