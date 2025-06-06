# 中文关键词提取工具

这是一个基于jieba分词和TF-IDF算法的中文关键词提取工具。该工具提供了两种关键词提取方法：
1. 使用jieba内置的TF-IDF接口
2. 使用自定义实现的TF-IDF算法

## 功能特点

- 支持中文文本的关键词提取
- 提供两种TF-IDF实现方式
- 支持自定义停用词
- 可配置返回的关键词数量
- 支持批量文本处理

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 基本用法：

```python
from keyword_extractor import KeywordExtractor

# 创建提取器实例
extractor = KeywordExtractor(topK=10)

# 使用jieba的TF-IDF接口
text = "你的文本内容"
keywords = extractor.extract_keywords_tfidf(text)

# 使用自定义TF-IDF实现
texts = ["文本1", "文本2", "文本3"]
keywords_list = extractor.extract_keywords_custom_tfidf(texts)
```

2. 添加停用词：
在 `data/stopwords.txt` 文件中添加停用词，每行一个词。

## 示例输出

```python
# 使用jieba的TF-IDF接口提取关键词：
人工智能: 0.4567
计算机科学: 0.2345
深度学习: 0.1234
...

# 使用自定义TF-IDF实现提取关键词：
人工智能: 0.4789
计算机科学: 0.2456
深度学习: 0.1345
...
```

## 注意事项

- 确保文本编码为UTF-8
- 停用词文件（stopwords.txt）是可选的
- 自定义TF-IDF实现更适合批量处理多个文档 