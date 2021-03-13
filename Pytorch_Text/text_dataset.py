import spacy
import pandas as pd
from torchtext.data import Field,BucketIterator,TabularDataset
from sklearn.model_selection import train_test_split

with  open("data/train_eng.txt") as eng_file:
    english_text = eng_file.read().split("\n")
with open("data/train_de.txt") as german_file:
    german_text = german_file.read().split("\n")

print(type(english_text))
#
raw_data = {
    "english":english_text[:60],
    "german":german_text[:60]
}

df = pd.DataFrame(raw_data,columns=["english","german"])
train,test = train_test_split(df,test_size=0.2)
train.to_json("data/train_lang.json",orient="records",lines=True)
test.to_json("data/test_lang.json",orient="records",lines=True)

train.to_csv("data/train.csv",index=False)
test.to_csv("data/test.csv",index=False)

spacy_eng = spacy.load("en_core_web_sm")
spacy_gem = spacy.load("de_core_news_sm")


def english_tokenizer(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]
def german_tokenizer(text):
    return [tok.text for tok in spacy_gem.tokenizer(text)]

english = Field(sequential=True,use_vocab=True,tokenize=english_tokenizer,lower=True)
german = Field(sequential=True,use_vocab=True,tokenize=german_tokenizer,lower=True)

fields = {"english":("eng",english),"german":("ger",german)}
train_data,test_data = TabularDataset.splits(
    path="",train="data/train_lang.json",test="data/test_lang.json",format="json",fields=fields
)

english.build_vocab(train_data,max_size=10000,min_freq=2)
german.build_vocab(train_data,max_size=10000,min_freq=2)

train_iterator,test_iterator = BucketIterator.splits(
    (train_data,test_data),batch_size=8,device="cuda"
)

for batch in train_iterator:
    print(batch)