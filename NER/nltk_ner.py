import nltk


def get_entities(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged)
    # 等价于
    # entities1 = nltk.ne_chunk(tagged, binary=True)
    entities_list = list(entities)
    triple = []
    for each_tuple in entities_list:
        if len(triple) >= 2:
            break
        if 'NNP' in each_tuple or 'NNPS' in each_tuple:
            triple.append(each_tuple[0])
        elif 'NN' in each_tuple or 'NNS' in each_tuple:
            triple.append(each_tuple[0])
    print('entitie_pairs: ', triple)
    return triple


if __name__ == '__main__':
    test = "The most common audits were about waste and recycling."
    get_entities(test)
