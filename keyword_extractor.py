import jieba
import jieba.analyse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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
        
        # 创建TF-IDF向量器
        vectorizer = TfidfVectorizer(
            max_features=self.topK,
            stop_words=self.stopwords
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

if __name__ == "__main__":
    main() 