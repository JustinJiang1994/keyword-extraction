import pandas as pd
from keyword_extractor import KeywordExtractor
import jieba

def extract_keywords_from_csv():
    """
    从CSV文件中抽取关键词
    """
    # 读取CSV文件
    try:
        df = pd.read_csv('data/technology_news.csv')
        print(f"成功读取CSV文件，共{len(df)}行数据")
        print(f"列名: {list(df.columns)}")
        
        # 检查content列是否存在
        if 'content' not in df.columns:
            print("错误：未找到'content'列")
            return
        
        # 合并所有文本内容
        all_text = ' '.join(df['content'].dropna().astype(str))
        print(f"合并后的文本长度: {len(all_text)} 字符")
        
        # 创建关键词提取器
        extractor = KeywordExtractor(topK=20)
        
        print("\n=== 使用jieba的TF-IDF接口提取关键词 ===")
        keywords_jieba = extractor.extract_keywords_tfidf(all_text)
        for i, (word, weight) in enumerate(keywords_jieba, 1):
            print(f"{i:2d}. {word}: {weight:.4f}")
        
        print("\n=== 使用自定义TF-IDF实现提取关键词 ===")
        # 将文本分割成多个段落进行处理
        texts = df['content'].dropna().astype(str).tolist()
        custom_keywords_list = extractor.extract_keywords_custom_tfidf(texts)
        
        # 合并所有文档的关键词并计算平均权重
        keyword_weights = {}
        for doc_keywords in custom_keywords_list:
            for word, weight in doc_keywords:
                if word in keyword_weights:
                    keyword_weights[word].append(weight)
                else:
                    keyword_weights[word] = [weight]
        
        # 计算平均权重并排序
        avg_keywords = [(word, sum(weights)/len(weights)) 
                       for word, weights in keyword_weights.items()]
        avg_keywords.sort(key=lambda x: x[1], reverse=True)
        
        for i, (word, weight) in enumerate(avg_keywords[:20], 1):
            print(f"{i:2d}. {word}: {weight:.4f}")
        
        print("\n=== 使用自定义TextRank实现提取关键词 ===")
        textrank_keywords = extractor.extract_keywords_textrank(all_text)
        for i, (word, weight) in enumerate(textrank_keywords, 1):
            print(f"{i:2d}. {word}: {weight:.4f}")
        
        print("\n=== 使用jieba的TextRank接口提取关键词 ===")
        jieba_textrank_keywords = extractor.extract_keywords_jieba_textrank(all_text)
        for i, (word, weight) in enumerate(jieba_textrank_keywords, 1):
            print(f"{i:2d}. {word}: {weight:.4f}")
        
        print("\n=== 使用LDA算法提取关键词 ===")
        # 使用前1000条新闻进行LDA分析，避免内存问题
        sample_texts = df['content'].dropna().astype(str).head(1000).tolist()
        lda_keywords = extractor.extract_keywords_lda(sample_texts, num_topics=10, num_words=20)
        for i, (word, weight) in enumerate(lda_keywords, 1):
            print(f"{i:2d}. {word}: {weight:.4f}")
            
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")

if __name__ == "__main__":
    extract_keywords_from_csv() 