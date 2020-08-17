"""
实现额外的方法
"""
import re
def tokenlize(sentence):
    """
    进行文本分词
    :param sentence:
    :return:[str, str, str]
    """
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“', ]
    sentence = sentence.lower() # 把大写转化为小写
    sentence = re.sub("<br />", " ", sentence)
    sentence = re.sub("|".join(fileters), " ", sentence)
    # sentence = re.sub("I'm","I am",sentence)
    # sentence = re.sub("isn't","is not",sentence)
    result = [i for i in sentence.split(" ") if len(i)>0]
    return result