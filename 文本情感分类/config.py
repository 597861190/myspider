"""
配置文件
"""
import pickle
train_batch_size = 128
test_batch_size = 500
ws = pickle.load(open("./models/ws.pkl", "rb")) #wb方式写入，rb的方式读
max_len = 50