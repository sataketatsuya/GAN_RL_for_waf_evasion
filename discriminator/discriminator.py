import torch
import torch.nn as nn
from transformers import BertModel


class Discriminator(nn.Module):
    def __init__(self, pretrained_model, dropout=0.1):
        super(Discriminator, self).__init__()

        self.model = BertModel.from_pretrained(pretrained_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 256)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False) # torch.Size([1, 768])
        dropout_output1 = self.dropout1(pooled_output) # torch.Size([1, 768])
        linear_output1 = self.linear1(dropout_output1) # torch.Size([1, 256])
        dropout_output2 = self.dropout2(linear_output1) # torch.Size([1, 256])
        linear_output2 = self.linear2(dropout_output2) # torch.Size([1, 1])
        validity = self.sigmoid(linear_output2) # torch.Size([1, 1])

        return validity
