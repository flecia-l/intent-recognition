import numpy as np

def load_glove_embeddings(file_path: str, embedding_dim: int) -> dict:
    """
    加载 GloVe 词向量到一个字典中
    :param file_path: GloVe 文件路径
    :param embedding_dim: 词向量的维度
    :return: 包含单词和对应词向量的字典
    """
    embeddings_index = {}
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector
    print(f"加载了 {len(embeddings_index)} 个词向量")
    return embeddings_index

def create_embedding_matrix(word_index: dict, embeddings_index: dict, embedding_dim: int) -> np.ndarray:
    """
    根据词汇表和 GloVe 词向量字典创建嵌入矩阵
    :param word_index: 词汇表索引
    :param embeddings_index: GloVe 词向量字典
    :param embedding_dim: 词向量维度
    :return: 嵌入矩阵
    """
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
