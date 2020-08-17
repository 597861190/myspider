"""

"""
from word_sequece import WordSequence
from dataset import get_dataloader
import pickle
from tqdm import tqdm
if __name__ == '__main__':
    ws = WordSequence()
    dl_train = get_dataloader(True)
    dl_test = get_dataloader(False)
    for review, label in tqdm(dl_train, total=len(dl_train)):
        for sentence in review:
            ws.fit(sentence)
    for review, label in tqdm(dl_train, total=len(dl_test)):
        for sentence in review:
            ws.fit(sentence)
    ws.build_vocab()
    print(len(ws))
    pickle.dump(ws, open("./models/ws.pkl", "wb"))