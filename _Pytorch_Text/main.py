import torch
import torchtext
import torch.nn as nn
import torch.optim as optim
import spacy
from torchtext.data import Field,TabularDataset,BucketIterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

spacy_en = spacy.load("en_core_web_sm") # Loading spacy language model

# tokenize_text = lambda x:x.split()  # A very simple way to tokenize text ,
def spacy_tokenize_text(text):
    return [tokenizer.text for tokenizer in spacy_en.tokenizer(text)]


d# and a binary label (0 or 1) denote whether the quotes is motivational or not

"""
The method "Field()" below contains the steps of operations needed to be done on the
"""
quote = Field(sequential=True,use_vocab=True,tokenize=spacy_tokenize_text,lower=True)
score = Field(seqeuntial=False,use_vocab=False)

# The below line is used to extract which columns from the original dataset are going to be use for example:
# In this case we are using both "quote" and "query"
fields = {"quote":('q',query),"score":('s',score)}

train_data,test_data = TabularDataset.splits(
    path="data",train="train.json",test="test.json",format="json",fields=fields
)

# # train_data, test_data = TabularDataset.splits(
# #                                         path='mydata',
# #                                         train='train.csv',
# #                                         test='test.csv',
# #                                         format='csv',
# #                                         fields=fields)

# # train_data, test_data = TabularDataset.splits(
# #                                         path='mydata',
# #                                         train='train.tsv',
# #                                         test='test.tsv',
# #                                         format='tsv',
# #                                         fields=fields)

quote.build_vocab(train_data,max_size=10000,min_freq=1,vectors="glove.6B.100d")


# Splitting between training and testing set
train_iterator,test_iterator = BucketIterator.splits(
    (train_data,test_data),batch_size=3,device=device
)

## SIMPLE LSTM MODEL
class RNN_LSTM(nn.Module):
    def __init__(self,input_size,embed_size,hidden_size,num_layers):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embedding = nn.Embedding(input_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers) # Specifying number of lstm units as num_layers
        self.fc1 =  nn.Linear(hidden_size,1) # One output (a single digit between 0 or 1)

    def forward(self,x):
        #setting initial hidden and cell states
        h0 = torch.zeros(self.num_layers,x.size(1),self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers,x.size(1),self.hidden_size).to(device)

        embedded = self.embedding(x)
        outputs ,_ = self.lstm(embedded,(h0,c0))
        predictions = self.fc_out(outputs[-1,:,:])
        return predictions


input_size = len(quote.build_vocab)
hidden_size = 512
num_layers = 2
embedding_size = 100
learning_rate = 0.05
num_epochs = 10

model = RNN_LSTM(input_size,embedding_size,hidden_size,num_layers).to(device)

pretrained_embeddings = quote.build_vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(num_epochs):
    for batch_idx,batch in enumerate(train_iterator):
        data = batch.q.to(device=device) # taking in batch of quotes
        targets = batch.s.to(device) # taking in batch of scores

        scores = model(data)
        loss = criterion(scores.sequeeze(1),targets.type_as(scores))

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

