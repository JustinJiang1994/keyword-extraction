import jieba
import jieba.analyse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import networkx as nx
from collections import defaultdict
import re
from gensim import corpora, models
from gensim.models import LdaModel
import warnings
warnings.filterwarnings('ignore')

class KeywordExtractor:
    def __init__(self, topK=20):
        """
        初始化关键词提取器
        :param topK: 返回的关键词数量
        """
        self.topK = topK
        # 加载停用词
        try:
            with open('data/stopwords.txt', 'r', encoding='utf-8') as f:
                self.stopwords = set([line.strip() for line in f])
        except FileNotFoundError:
            print("警告：未找到停用词文件，将使用空停用词列表")
            self.stopwords = set()

    def extract_keywords_tfidf(self, text):
        """
        使用TF-IDF算法提取关键词
        :param text: 输入文本
        :return: 关键词列表及其权重
        """
        # 使用jieba的TF-IDF接口
        keywords = jieba.analyse.extract_tags(text, 
                                            topK=self.topK,
                                            withWeight=True,
                                            allowPOS=('n', 'vn', 'v', 'a', 'an'))
        return keywords

    def extract_keywords_custom_tfidf(self, texts):
        """
        使用自定义TF-IDF实现提取关键词
        :param texts: 文本列表
        :return: 每个文本的关键词及其权重
        """
        # 对文本进行分词
        segmented_texts = [' '.join(jieba.cut(text)) for text in texts]
        
        # 创建TF-IDF向量器，使用jieba分词后的停用词
        # 将中文停用词转换为分词后的形式
        jieba_stopwords = set()
        for stopword in self.stopwords:
            if len(stopword) > 1:  # 只处理长度大于1的停用词
                jieba_stopwords.add(stopword)
        
        vectorizer = TfidfVectorizer(
            max_features=self.topK,
            stop_words=list(jieba_stopwords) if jieba_stopwords else None
        )
        
        # 计算TF-IDF矩阵
        tfidf_matrix = vectorizer.fit_transform(segmented_texts)
        
        # 获取特征词
        feature_names = vectorizer.get_feature_names_out()
        
        # 为每个文档提取关键词
        results = []
        for i in range(len(texts)):
            # 获取当前文档的TF-IDF值
            tfidf_scores = tfidf_matrix[i].toarray()[0]
            # 创建(词, 权重)对
            keywords = [(feature_names[j], tfidf_scores[j]) 
                       for j in range(len(feature_names)) 
                       if tfidf_scores[j] > 0]
            # 按权重排序
            keywords.sort(key=lambda x: x[1], reverse=True)
            results.append(keywords[:self.topK])
            
        return results

    def extract_keywords_textrank(self, text, window_size=4, damping=0.85, max_iter=100):
        """
        使用TextRank算法提取关键词
        :param text: 输入文本
        :param window_size: 窗口大小，用于构建词图
        :param damping: 阻尼系数
        :param max_iter: 最大迭代次数
        :return: 关键词列表及其权重
        """
        # 使用jieba进行分词和词性标注
        words = jieba.posseg.cut(text)
        
        # 过滤停用词和短词，只保留名词、动词、形容词等
        allowed_pos = {'n', 'nr', 'ns', 'nt', 'nz', 'vn', 'v', 'a', 'an'}
        filtered_words = []
        
        for word, pos in words:
            if (len(word) > 1 and 
                word not in self.stopwords and 
                pos in allowed_pos):
                filtered_words.append(word)
        
        if len(filtered_words) < 2:
            return []
        
        # 构建词图
        word_graph = defaultdict(list)
        word_freq = defaultdict(int)
        
        # 统计词频
        for word in filtered_words:
            word_freq[word] += 1
        
        # 构建共现图
        for i in range(len(filtered_words)):
            for j in range(i + 1, min(i + window_size + 1, len(filtered_words))):
                word1, word2 = filtered_words[i], filtered_words[j]
                if word1 != word2:
                    word_graph[word1].append(word2)
                    word_graph[word2].append(word1)
        
        # 创建NetworkX图
        G = nx.Graph()
        
        # 添加节点
        for word in word_freq:
            G.add_node(word, weight=word_freq[word])
        
        # 添加边
        for word, neighbors in word_graph.items():
            for neighbor in neighbors:
                if G.has_node(neighbor):
                    # 边的权重基于共现次数
                    weight = word_graph[word].count(neighbor)
                    G.add_edge(word, neighbor, weight=weight)
        
        # 使用PageRank算法计算节点重要性
        try:
            scores = nx.pagerank(G, alpha=damping, max_iter=max_iter)
        except nx.PowerIterationFailedConvergence:
            # 如果PageRank不收敛，使用度中心性
            scores = nx.degree_centrality(G)
        
        # 按分数排序并返回topK个关键词
        keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return keywords[:self.topK]

    def extract_keywords_jieba_textrank(self, text):
        """
        使用jieba内置的TextRank接口提取关键词
        :param text: 输入文本
        :return: 关键词列表及其权重
        """
        # 使用jieba的TextRank接口
        keywords = jieba.analyse.textrank(text, 
                                        topK=self.topK,
                                        withWeight=True,
                                        allowPOS=('n', 'vn', 'v', 'a', 'an'))
        return keywords

    def extract_keywords_lda(self, texts, num_topics=10, num_words=20, passes=10):
        """
        使用LDA算法提取关键词
        :param texts: 文本列表
        :param num_topics: 主题数量
        :param num_words: 每个主题的词数
        :param passes: 训练轮数
        :return: 关键词列表及其权重
        """
        # 对文本进行分词
        processed_texts = []
        for text in texts:
            words = jieba.cut(text)
            # 过滤停用词和短词
            filtered_words = [word for word in words 
                            if len(word) > 1 and word not in self.stopwords]
            processed_texts.append(filtered_words)
        
        if not processed_texts or len(processed_texts) < 2:
            return []
        
        # 创建词典
        dictionary = corpora.Dictionary(processed_texts)
        
        # 过滤极端频率的词
        dictionary.filter_extremes(no_below=2, no_above=0.5)
        
        if len(dictionary) == 0:
            return []
        
        # 创建文档-词频矩阵
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        
        # 训练LDA模型
        try:
            lda_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=min(num_topics, len(dictionary)),
                random_state=42,
                passes=passes,
                alpha='auto',
                per_word_topics=True
            )
        except Exception as e:
            print(f"LDA模型训练失败: {e}")
            return []
        
        # 提取所有主题的关键词
        all_keywords = {}
        
        for topic_id in range(lda_model.num_topics):
            topic_words = lda_model.show_topic(topic_id, num_words)
            for word, weight in topic_words:
                if word in all_keywords:
                    all_keywords[word] = max(all_keywords[word], weight)
                else:
                    all_keywords[word] = weight
        
        # 按权重排序并返回topK个关键词
        keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)
        return keywords[:self.topK]

    def extract_keywords_lda_single_text(self, text, num_topics=5, num_words=20, passes=10):
        """
        对单个文本使用LDA算法提取关键词
        :param text: 输入文本
        :param num_topics: 主题数量
        :param num_words: 每个主题的词数
        :param passes: 训练轮数
        :return: 关键词列表及其权重
        """
        # 将单个文本分割成多个段落
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        if len(paragraphs) < 2:
            # 如果段落太少，按句子分割
            import re
            sentences = re.split(r'[。！？；]', text)
            paragraphs = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(paragraphs) < 2:
            # 如果还是太少，直接返回空
            return []
        
        return self.extract_keywords_lda(paragraphs, num_topics, num_words, passes)

def main():
    # 示例用法
    extractor = KeywordExtractor(topK=10)
    
    # 示例文本
    text = """
    人工智能是计算机科学的一个分支，它致力于研究和开发能够模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。
    人工智能的主要研究领域包括机器学习、深度学习、自然语言处理、计算机视觉等。近年来，随着大数据和计算能力的提升，
    人工智能技术取得了突飞猛进的发展，在医疗、金融、教育等多个领域都得到了广泛应用。
    """
    
    print("使用jieba的TF-IDF接口提取关键词：")
    keywords = extractor.extract_keywords_tfidf(text)
    for word, weight in keywords:
        print(f"{word}: {weight:.4f}")
    
    print("\n使用自定义TF-IDF实现提取关键词：")
    texts = [text]  # 示例中只有一个文本
    custom_keywords = extractor.extract_keywords_custom_tfidf(texts)[0]
    for word, weight in custom_keywords:
        print(f"{word}: {weight:.4f}")
    
    print("\n使用自定义TextRank实现提取关键词：")
    textrank_keywords = extractor.extract_keywords_textrank(text)
    for word, weight in textrank_keywords:
        print(f"{word}: {weight:.4f}")
    
    print("\n使用jieba的TextRank接口提取关键词：")
    jieba_textrank_keywords = extractor.extract_keywords_jieba_textrank(text)
    for word, weight in jieba_textrank_keywords:
        print(f"{word}: {weight:.4f}")
    
    print("\n使用LDA算法提取关键词：")
    lda_keywords = extractor.extract_keywords_lda_single_text(text)
    for word, weight in lda_keywords:
        print(f"{word}: {weight:.4f}")

if __name__ == "__main__":
    main() 