"""获取AI/算法论文"""
import os
import re
import math
import traceback
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from collections import defaultdict
from ai_categories_config import CATEGORY_DISPLAY_ORDER, CATEGORY_THRESHOLDS, CATEGORY_KEYWORDS
from llm_helper import ChatGLMHelper
from typing import Dict, List, Tuple, Optional
import traceback
import arxiv

# 查询参数设置
QUERY_DAYS_AGO = 1          # 查询几天前的论文，0=今天，1=昨天，2=前天
MAX_RESULTS = 600           # 最大返回论文数量
MAX_WORKERS = 2            # 并行处理的最大线程数

# ArXiv 类别配置（核心AI类别）
ARXIV_CATEGORIES = [
    "cs.AI",    # Artificial Intelligence
    "cs.CL",    # Computation and Language (NLP)
    "cs.CV",    # Computer Vision
    "cs.LG",    # Machine Learning
    "stat.ML",  # Statistics - Machine Learning
    "cs.NE",    # Neural and Evolutionary Computing
]

# 导入NLTK库用于文本预处理
try:
    import nltk
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    
    # 创建标志文件路径
    nltk_flag_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.nltk_data_downloaded')
    
    # 检查是否已经下载过NLTK数据
    if os.path.exists(nltk_flag_file):
        # 已经下载过，直接使用
        NLTK_AVAILABLE = True
    else:
        # 检查必要的NLTK数据是否已下载
        needed_data = []
        for data_name in ['punkt', 'wordnet', 'stopwords']:
            try:
                path = f"{'tokenizers/' if data_name == 'punkt' else 'corpora/'}{data_name}"
                nltk.data.find(path)
                print(f"NLTK数据 '{data_name}' 已存在于: {path}")
            except LookupError:
                needed_data.append(data_name)
                print(f"NLTK数据 '{data_name}' 不存在，需要下载")
        
        # 只下载缺失的数据
        if needed_data:
            print(f"正在下载缺失的NLTK数据文件: {', '.join(needed_data)}")
            for data_name in needed_data:
                print(f"开始下载 '{data_name}'...")
                download_result = nltk.download(data_name, quiet=False)
                print(f"下载 '{data_name}' 结果: {download_result}")
            print("NLTK数据文件下载完成")
        
        # 特别处理punkt_tab
        try:
            nltk.data.find('tokenizers/punkt_tab')
            print("NLTK数据 'punkt_tab' 已存在")
        except LookupError:
            print("开始下载 'punkt_tab'...")
            download_result = nltk.download('punkt', quiet=False)  # 重新下载 punkt可能会包含punkt_tab
            print(f"下载 'punkt' 结果: {download_result}")
        
        # 创建标志文件表示数据已下载
        with open(nltk_flag_file, 'w') as f:
            f.write(f"NLTK data downloaded at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        NLTK_AVAILABLE = True
    
    NLTK_AVAILABLE = True
except ImportError:
    print("NLTK库未安装，将使用基本文本处理")
    NLTK_AVAILABLE = False

def extract_github_link(paper):
    """从论文中提取代码链接（GitHub、项目主页等）

    Args:
        paper: arXiv论文对象

    Returns:
        str: 代码链接或None
    """
    # GitHub链接模式
    github_patterns = [
        # GitHub链接
        r'https?://github\.com/[a-zA-Z0-9-]+/[a-zA-Z0-9-_.]+',
        r'github\.com/[a-zA-Z0-9-]+/[a-zA-Z0-9-_.]+',
        r'https?://www\.github\.com/[a-zA-Z0-9-]+/[a-zA-Z0-9-_.]+',
        r'www\.github\.com/[a-zA-Z0-9-]+/[a-zA-Z0-9-_.]+',
        # 项目页面
        r'https?://[a-zA-Z0-9-]+\.github\.io/[a-zA-Z0-9-_.]+',
        # 通用代码链接模式
        r'code.*available.*?(?:https?://github\.com/[^\s<>"]+)',
        r'implementation.*?(?:https?://github\.com/[^\s<>"]+)',
        r'source.*code.*?(?:https?://github\.com/[^\s<>"]+)',
    ]

    # 要搜索的文本来源
    text_sources = []
    
    # 1. 摘要
    if hasattr(paper, 'summary') and paper.summary:
        text_sources.append(paper.summary)
    
    # 2. 评论
    if hasattr(paper, 'comments') and paper.comments:
        text_sources.append(paper.comments)
    
    # 3. 期刊引用
    if hasattr(paper, 'journal_ref') and paper.journal_ref:
        text_sources.append(paper.journal_ref)
    
    # 4. 链接列表
    if hasattr(paper, 'links'):
        for link in paper.links:
            if hasattr(link, 'href') and link.href:
                text_sources.append(link.href)
    
    # 5. DOI
    if hasattr(paper, 'doi') and paper.doi:
        text_sources.append(paper.doi)
    
    # 从所有文本来源中查找GitHub链接
    for text in text_sources:
        for pattern in github_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                url = match.group(0)
                if not url.startswith('http'):
                    url = 'https://' + url
                return url
    
    # 如果没有找到GitHub链接，尝试从links中提取项目主页
    if hasattr(paper, 'links'):
        for link in paper.links:
            if hasattr(link, 'href') and link.href:
                href = link.href
                # 检查是否是项目主页（非arXiv、非PDF）
                if (href and 
                    'arxiv.org' not in href.lower() and 
                    'pdf' not in href.lower() and
                    ('http://' in href or 'https://' in href)):
                    # 检查是否包含常见代码仓库关键词
                    code_keywords = ['code', 'github', 'gitlab', 'bitbucket', 'project', 'demo', 'page']
                    if any(keyword in href.lower() for keyword in code_keywords):
                        return href
    
    return None


def extract_arxiv_id(url):
    """从ArXiv URL中提取论文ID

    Args:
        url: ArXiv论文URL

    Returns:
        str: 论文ID
    """
    # 处理不同格式的ArXiv URL
    patterns = [
        r"arxiv\.org/abs/(\d+\.\d+)",
        r"arxiv\.org/pdf/(\d+\.\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def df_to_markdown_table(papers_by_category: dict, target_date) -> str:
    """生成表格形式的Markdown内容，支持两级类别标题"""
    markdown = ""
    
    # 过滤掉没有论文的类别
    active_categories = {k: v for k, v in papers_by_category.items() if v}
    
    if not active_categories:
        return "今天没有相关论文。"
    
    # 表格列标题
    headers = ['状态', '英文标题', '中文标题', '作者', 'PDF链接', '代码/贡献']
    
    # 按照CATEGORY_DISPLAY_ORDER的顺序处理类别
    for category in CATEGORY_DISPLAY_ORDER:
        if category not in active_categories:
            continue
        # 只输出一次主类别标题
        markdown += f"\n## {category}\n\n"
        papers_by_subcategory = defaultdict(list)
        for paper in active_categories[category]:
            subcategory = paper.get('subcategory', '')
            papers_by_subcategory[subcategory].append(paper)
        if not papers_by_subcategory:
            continue
        for subcategory, papers in papers_by_subcategory.items():
            markdown += f"\n### {subcategory}\n\n"
            markdown += "|" + "|".join(headers) + "|\n"
            markdown += "|" + "|".join(["---"] * len(headers)) + "|\n"
            for paper in papers:
                if paper['is_updated']:
                    status = f"📝 更新"
                else:
                    status = f"🆕 发布"
                def summarize_contribution(core_contribution):
                    if not core_contribution:
                        return []
                    if "|" in core_contribution:
                        items = [item.strip() for item in core_contribution.split("|")]
                    else:
                        items = [core_contribution.strip()]
                    blacklist = ["代码开源", "提供数据集", "代码已开源", "数据集已公开"]
                    items = [i for i in items if all(b not in i for b in blacklist)]
                    items = items[:2]
                    items = [(i[:50] + ("..." if len(i) > 50 else "")) for i in items]
                    return items
                contrib_list = []
                if "核心贡献" in paper:
                    contrib_list = summarize_contribution(paper["核心贡献"])
                if paper['github_url'] != 'None':
                    code_and_contribution = f"[代码]({paper['github_url']})"
                    if contrib_list:
                        code_and_contribution += "; " + "; ".join(contrib_list)
                elif contrib_list:
                    code_and_contribution = "; ".join(contrib_list)
                else:
                    code_and_contribution = '无'
                values = [
                    status,
                    paper['title'],
                    paper.get('title_zh', ''),
                    paper['authors'],
                    f"[PDF]({paper['pdf_url']})",
                    code_and_contribution,
                ]
                values = [str(v).replace('\n', ' ').replace('|', '&#124;') for v in values]
                markdown += "|" + "|".join(values) + "|\n"
            markdown += "\n"
    return markdown


def df_to_markdown_detailed(papers_by_category: dict, target_date) -> str:
    """生成详细格式的Markdown内容，支持两级类别标题"""
    markdown = ""
    
    # 过滤掉没有论文的类别
    active_categories = {k: v for k, v in papers_by_category.items() if v}
    
    if not active_categories:
        return "今天没有相关论文。"
    
    # 按照CATEGORY_DISPLAY_ORDER的顺序处理类别
    for category in CATEGORY_DISPLAY_ORDER:
        if category not in active_categories:
            continue
            
        # 添加一级类别标题
        markdown += f"\n## {category}\n\n"
        
        # 按子类别组织论文
        papers_by_subcategory = defaultdict(list)
        
        # 将所有论文分配到子类别
        for paper in active_categories[category]:
            subcategory = paper.get('subcategory', '')
            papers_by_subcategory[subcategory].append(paper)
        
        # 如果当前类别下没有论文，跳过
        if not papers_by_subcategory:
            continue
            
        # 处理每个子类别
        for subcategory, papers in papers_by_subcategory.items():
            # 添加二级类别标题
            markdown += f"\n### {subcategory}\n\n"
            
            # 添加论文详细信息
            for idx, paper in enumerate(papers, 1):
                # 引用编号
                markdown += f'**index:** {idx}<br />\n'
                # 日期
                markdown += f'**Date:** {target_date.strftime("%Y-%m-%d")}<br />\n'
                # 英文标题
                markdown += f'**Title:** {paper["title"]}<br />\n'
                # 中文标题
                markdown += f'**Title_cn:** {paper.get("title_zh", "")}<br />\n'
                # 作者（已经是格式化好的字符串）
                markdown += f'**Authors:** {paper["authors"]}<br />\n'
                # PDF链接
                markdown += f'**PDF:** [PDF]({paper["pdf_url"]})<br />\n'

                # 合并代码链接和精简后的核心贡献
                markdown += '**Code/Contribution:**\n'
                
                # 精简核心贡献内容
                def summarize_contribution(core_contribution):
                    if not core_contribution:
                        return []
                    # 分割为多条
                    if "|" in core_contribution:
                        items = [item.strip() for item in core_contribution.split("|")] 
                    else:
                        items = [core_contribution.strip()]
                    # 去除模板化内容
                    blacklist = ["代码开源", "提供数据集", "代码已开源", "数据集已公开"]
                    items = [i for i in items if all(b not in i for b in blacklist)]
                    # 只保留前三条
                    items = items[:3]
                    return items
                
                # 处理核心贡献
                contrib_list = []
                if "核心问题" in paper:
                    markdown += f'问题：{paper["核心问题"]}\n'
                
                if "核心方法" in paper:
                    markdown += f'方法：{paper["核心方法"]}\n'
                
                if "核心贡献" in paper:
                    contrib_list = summarize_contribution(paper["核心贡献"])
                    if contrib_list:
                        markdown += f'{", ".join(contrib_list)}\n'
                
                # 处理代码链接
                if paper['github_url'] != 'None':
                    markdown += f'[代码]({paper["github_url"]})\n'
                
                # 添加空行
                markdown += '\n'

    return markdown


def preprocess_text(text: str) -> str:
    """
    对文本进行预处理，包括小写转换、分词、去停用词、词干提取和词形还原
    
    Args:
        text: 原始文本
        
    Returns:
        str: 预处理后的文本
    """
    # 转换为小写
    text = text.lower()
    
    # 基本文本处理：先去除特殊字符
    basic_processed = re.sub(r'[^\w\s]', ' ', text)
    
    # 如果NLTK不可用，直接返回基本处理结果
    if not NLTK_AVAILABLE:
        return basic_processed
    
    # 尝试使用NLTK进行高级处理
    try:
        # 分词 - 先使用基本分词作为备选
        try:
            tokens = word_tokenize(text)
        except Exception:
            # 如果高级分词失败，使用基本分词
            tokens = basic_processed.split()
        
        # 去除停用词
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        except Exception:
            # 如果停用词处理失败，使用基本停用词列表
            basic_stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'for', 'to', 'of', 'and', 'or', 'with', 'by'}
            tokens = [token for token in tokens if token not in basic_stop_words and len(token) > 2]
        
        # 词干提取和词形还原 - 可选功能
        try:
            stemmer = PorterStemmer()
            stemmed_tokens = [stemmer.stem(token) for token in tokens]
            
            lemmatizer = WordNetLemmatizer()
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]
            
            # 重新组合成文本
            return " ".join(lemmatized_tokens)
        except Exception:
            # 如果词干提取或词形还原失败，只返回分词和去停用词的结果
            return " ".join(tokens)
    
    except Exception as e:
        print(f"NLTK处理文本时出错: {str(e)}")
        # 如果所有NLTK处理都失败，回退到基本处理
        return basic_processed


def get_category_by_keywords(title: str, abstract: str, categories_config: Dict) -> List[Tuple[str, float, Optional[Tuple[str, float]], Optional[Dict]]]:
    """
    执行基于关键词匹配和优先级规则的层次化论文分类，带有增强的文本处理和置信度评分。
    
    Args:
        title (str): 论文标题，用于主要上下文分析
        abstract (str): 论文摘要，用于全面内容分析
        categories_config (Dict): 包含类别定义、关键词、权重和优先级的配置字典
    
    实现细节:
        1. 增强文本预处理:
           - 大小写标准化和标准化处理
           - 标题和摘要的组合分析，使用差异化权重
           - 高级分词和停用词过滤
           - 多级词干提取和词形还原
           - N-gram分析，提高短语匹配准确性
        
        2. 优化评分机制:
           - 主要得分: 加权关键词匹配 (动态基础权重)
           - 标题加成: 标题匹配的额外权重 (优化加权)
           - 精确匹配加成: 完整短语匹配的额外权重
           - 优先级乘数: 类别特定重要性缩放
           - 负面关键词惩罚: 使用改进的逻辑函数平滑惩罚
           - 类别相关性判断: 考虑类别间的相关性
        
        3. 智能分类逻辑:
           - 使用类别自定义阈值与动态阈值调整
           - 增强的子类别分类
           - 优先类别的层次化处理
           - 智能回退机制，考虑类别相关性
           - 置信度评分和分类解释
    
    Returns:
        List[Tuple[str, float, Optional[Tuple[str, float]], Optional[Dict]]]: 按置信度降序排序的 
        (类别, 置信度分数, 子类别信息, 分类解释) 元组列表
    """
    # 文本预处理
    title_lower = title.lower()
    abstract_lower = abstract.lower()
    
    # 使用高级文本预处理
    processed_title = preprocess_text(title)
    processed_abstract = preprocess_text(abstract)
    processed_combined = processed_title + " " + processed_abstract
    
    # 移除常见的停用词，提高匹配质量
    stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'for', 'to', 'of', 'and', 'or', 'with', 'by', 
                 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 
                 'does', 'did', 'but', 'if', 'then', 'else', 'when', 'up', 'down', 'this', 'that'}
    
    # 分词并过滤停用词
    title_words = set(w for w in title_lower.split() if w not in stop_words)
    abstract_words = set(w for w in abstract_lower.split() if w not in stop_words)
    
    # 组合文本用于匹配
    combined_text = title_lower + " " + abstract_lower
    
    # 初始化得分累加器和匹配记录
    scores = defaultdict(float)
    match_details = defaultdict(list)
    
    # 计算每个类别的得分
    for category, config in categories_config.items():
        score = 0.0
        matches = []
        
        # 1. 正向关键词匹配
        for keyword, weight in config["keywords"]:
            keyword_lower = keyword.lower()
            keyword_words = set(w for w in keyword_lower.split() if w not in stop_words)
            
            # 对关键词也进行预处理
            processed_keyword = preprocess_text(keyword)
            
            # 完整短语精确匹配（最高权重）
            if keyword_lower in title_lower:
                match_score = weight * 0.25  # 标题中的精确匹配权重最高
                score += match_score
                matches.append(f"标题精确匹配 [{keyword}]: +{match_score:.2f}")
            elif keyword_lower in abstract_lower:
                match_score = weight * 0.15  # 摘要中的精确匹配权重次之
                score += match_score
                matches.append(f"摘要精确匹配 [{keyword}]: +{match_score:.2f}")
            
            # 使用预处理后的文本进行匹配（提高准确性）
            elif processed_keyword in processed_title:
                match_score = weight * 0.22  # 预处理标题中的匹配权重高
                score += match_score
                matches.append(f"标题语义匹配 [{keyword}]: +{match_score:.2f}")
            elif processed_keyword in processed_abstract:
                match_score = weight * 0.14  # 预处理摘要中的匹配权重中等
                score += match_score
                matches.append(f"摘要语义匹配 [{keyword}]: +{match_score:.2f}")
            
            # 标题中的关键词组合匹配（高权重）
            elif len(keyword_words) > 1 and keyword_words.issubset(title_words):
                match_score = weight * 0.18  # 标题中的词组匹配权重高
                score += match_score
                matches.append(f"标题词组匹配 [{keyword}]: +{match_score:.2f}")
            
            # 摘要中的关键词组合匹配（中等权重）
            elif len(keyword_words) > 1 and keyword_words.issubset(abstract_words):
                match_score = weight * 0.12  # 摘要中的词组匹配权重中等
                score += match_score
                matches.append(f"摘要词组匹配 [{keyword}]: +{match_score:.2f}")
            
            # 单词匹配（低权重）
            else:
                # 将关键词拆分为单词进行匹配
                word_matches = 0
                title_match_bonus = 0
                
                # 分别处理原始文本和预处理文本的匹配
                for word in keyword_words:
                    if len(word) <= 3:  # 忽略过短的词
                        continue
                        
                    if word in title_words:
                        word_matches += 1
                        title_match_bonus += 1  # 标题匹配额外加分
                    elif word in abstract_words:
                        word_matches += 0.6  # 摘要匹配的权重低于标题
                
                # 处理预处理文本中的匹配
                processed_keyword_words = processed_keyword.split()
                for word in processed_keyword_words:
                    if len(word) <= 3:
                        continue
                    if word in processed_title:
                        word_matches += 0.5
                        title_match_bonus += 0.3
                    elif word in processed_abstract:
                        word_matches += 0.3
                
                # 如果有单词匹配，计算得分
                if word_matches > 0:
                    # 计算匹配比例
                    match_ratio = word_matches / len(keyword_words)
                    # 基础得分
                    base_score = weight * match_ratio * 0.08
                    # 标题匹配加成
                    bonus_score = title_match_bonus * 0.02
                    # 总得分
                    match_score = base_score + bonus_score
                    score += match_score
                    matches.append(f"单词匹配 [{keyword}]: +{match_score:.2f} (匹配度: {match_ratio:.1%})")
        
        # 2. 负向关键词惩罚
        negative_score = 0.0
        for neg_keyword, neg_weight in config.get("negative_keywords", []):
            neg_keyword_lower = neg_keyword.lower()
            neg_keyword_words = set(w for w in neg_keyword_lower.split() if w not in stop_words)
            
            # 完整短语匹配（严重惩罚）
            if neg_keyword_lower in title_lower:
                negative_score += neg_weight * 1.0  # 标题中的负向关键词严重惩罚
                matches.append(f"负向关键词 [{neg_keyword}] 在标题中: -{neg_weight:.2f}")
            elif neg_keyword_lower in abstract_lower:
                negative_score += neg_weight * 0.7  # 摘要中的负向关键词中度惩罚
                matches.append(f"负向关键词 [{neg_keyword}] 在摘要中: -{neg_weight * 0.7:.2f}")
            
            # 关键词组合匹配（中度惩罚）
            elif len(neg_keyword_words) > 1 and neg_keyword_words.issubset(title_words):
                negative_score += neg_weight * 0.8
                matches.append(f"负向词组 [{neg_keyword}] 在标题中: -{neg_weight * 0.8:.2f}")
            elif len(neg_keyword_words) > 1 and neg_keyword_words.issubset(abstract_words):
                negative_score += neg_weight * 0.5
                matches.append(f"负向词组 [{neg_keyword}] 在摘要中: -{neg_weight * 0.5:.2f}")
        
        # 应用负向关键词惩罚
        if negative_score > 0:
            # 使用指数衰减进行惩罚
            score *= math.exp(-negative_score)
        
        # 3. 优先级调整
        priority = config.get("priority", 1.0)
        if priority > 1.0:
            # 优先级乘数：高优先级类别获得额外加成
            priority_multiplier = 1.0 + (priority - 1.0) * 0.1
            score *= priority_multiplier
        
        # 记录得分和匹配信息
        if score > 0:
            scores[category] = score
            match_details[category] = matches
    
    # 4. 类别选择和置信度计算
    if not scores:
        return []
    
    # 获取最高得分
    max_score = max(scores.values())
    
    # 筛选候选类别（使用配置文件中的阈值）
    candidate_categories = []
    for category, score in scores.items():
        # 获取该类别的阈值配置
        threshold_config = CATEGORY_THRESHOLDS.get(category, {})
        threshold = threshold_config.get("threshold", 1.0)
        
        # 如果得分超过阈值，则加入候选类别
        if score >= threshold:
            candidate_categories.append((category, score))
    
    # 如果没有候选类别，返回空列表（略过该论文）
    if not candidate_categories:
        return []
    
    # 按得分降序排序
    candidate_categories.sort(key=lambda x: x[1], reverse=True)
    
    # 5. 子类别分类
    subcategory_info = None
    if candidate_categories:
        best_category = candidate_categories[0][0]
        subcategory_info = classify_subcategory(title, abstract, best_category, categories_config)
    
    # 6. 构建分类解释
    classification_explanation = {
        "max_score": max_score,
        "total_categories": len(candidate_categories),
        "top_matches": [
            {
                "category": cat,
                "score": score,
                "details": match_details.get(cat, [])
            }
            for cat, score in candidate_categories[:3]
        ]
    }
    
    # 7. 返回结果
    result = []
    for category, score in candidate_categories:
        result.append((category, score, subcategory_info if category == candidate_categories[0][0] else None, classification_explanation if category == candidate_categories[0][0] else None))
    
    return result


def classify_subcategory(title: str, abstract: str, main_category: str, categories_config: Dict) -> Tuple[str, float]:
    """
    对论文进行子类别分类
    
    Args:
        title: 论文标题
        abstract: 论文摘要
        main_category: 主类别
        categories_config: 类别配置
    
    Returns:
        Tuple[str, float]: (子类别名称, 置信度)
    """
    # 获取主类别的子类别配置（从CATEGORY_THRESHOLDS获取）
    from ai_categories_config import CATEGORY_THRESHOLDS, CATEGORY_KEYWORDS
    main_config = CATEGORY_THRESHOLDS.get(main_category, {})
    subcategories = main_config.get("subcategories", {})
    
    if not subcategories:
        return (None, 0.0)
    
    # 使用子类别关键词进行匹配
    title_lower = title.lower()
    abstract_lower = abstract.lower()
    
    # 为每个子类别定义关键词（从CATEGORY_KEYWORDS中提取相关关键词）
    subcategory_keywords = {
        # AI Agents 子类别关键词
        "单代理规划与工具使用 (Single Agent Planning & Tool Use)": ["autonomous agent", "tool use", "ReAct", "reflexion", "agent planning", "tool-using", "function calling", "API calling", "tool calling", "single agent", "individual agent"],
        "多代理协作系统 (Multi-Agent Collaboration)": ["multi-agent system", "multi-agent collaboration", "agent society", "emergent behavior", "multi-agent", "agent interaction", "agent cooperation", "agent coordination", "agent team", "cooperative agent"],
        "长链推理与思考链 (Long-Chain Reasoning & CoT)": ["chain-of-thought", "o1-like", "test-time compute", "long-chain reasoning", "reasoning model", "CoT", "reasoning capability", "complex reasoning", "logical reasoning"],
        "上下文工程 (Context Engineering)": ["context engineering", "agentic context", "dynamic context", "context optimization", "in-context management", "context management", "context compression", "RAG"],
        "Agent评估与基准 (Agent Evaluation & Benchmarks)": ["agent benchmark", "agent evaluation", "GAIA", "WebArena", "agent performance", "agent testing", "agent test", "agent metric"],
        "Agentic Workflow与自动化 (Agentic Workflow & Automation)": ["agentic workflow", "agent orchestration", "long-horizon task", "task decomposition", "workflow automation", "autonomous execution", "agentic system", "agent workflow", "autonomous workflow"],

        # 多模态模型 子类别关键词
        "视觉-语言模型 (Vision-Language Models, VLM)": ["vision-language model", "VLM", "image-text alignment", "visual question answering", "vision-language", "visual-language", "multimodal transformer", "visual-language model", "image-text"],
        "视频与时序多模态 (Video & Temporal Multimodal)": ["video understanding", "video-language", "temporal multimodal", "long video model", "video captioning", "video-text", "video generation", "action recognition", "video analysis"],
        "音频-视觉-文本融合 (Audio-Visual-Text Fusion)": ["audio-visual", "speech multimodal", "audio-language model", "audio-text", "speech-language", "audio event", "music generation", "multimodal audio"],
        "3D/4D与空间多模态 (3D/4D & Spatial Multimodal)": ["3D multimodal", "4D generation", "gaussian splatting", "spatial understanding", "3D reconstruction", "neural rendering", "NeRF", "3D vision", "point cloud", "spatial multimodal"],
        "生成式多模态 (Generative Multimodal)": ["generative multimodal", "diffusion model multimodal", "generative AI", "AIGC", "multimodal generation", "text-to-image", "text-to-video", "image generation", "multimodal generation"],
        "统一多模态预训练 (Unified Multimodal Pretraining)": ["unified multimodal", "any-to-any", "multimodal foundation model", "omnimodal", "multimodal pretraining", "cross-modal pretraining", "unified model", "multimodal pretraining"],

        # 大模型高效训练与推理 子类别关键词
        "模型压缩与量化 (Model Compression & Quantization)": ["model compression", "quantization", "pruning", "model quantization", "post-training quantization", "quantization-aware training", "network pruning", "model pruning", "low-rank adaptation"],
        "推理加速技术 (Inference Acceleration)": ["inference optimization", "speculative decoding", "flash attention", "KV cache", "inference acceleration", "fast inference", "inference speedup", "KV cache optimization", "inference speed"],
        "混合专家模型 (Mixture of Experts, MoE)": ["mixture of experts", "MoE", "sparse MoE", "expert routing", "sparse expert", "mixture-of-experts"],
        "合成数据生成 (Synthetic Data Generation)": ["synthetic data generation", "data distillation", "self-reward data", "synthetic data", "data augmentation", "self-supervised data", "synthetic training"],
        "能效与可持续训练 (Energy Efficiency & Sustainable Training)": ["energy efficient AI", "green AI", "carbon aware computing", "hardware-aware training", "low-power training", "energy optimization", "sustainable training", "carbon footprint", "energy consumption"],

        # 具身智能与机器人 子类别关键词
        "机器人学习基础 (Robot Learning Foundations)": ["robot learning", "reinforcement learning robotics", "imitation learning robot", "robotic learning", "robotics", "robot control", "policy learning", "robot policy"],
        "仿真到现实迁移 (Sim-to-Real Transfer)": ["sim-to-real", "domain randomization", "sim2real transfer", "simulation to reality", "domain adaptation robotics", "real-world robot", "sim2real"],
        "世界模型与预测 (World Model & Prediction)": ["world model", "video prediction robotics", "physical reasoning", "world modeling", "environment model", "physics-based", "predictive model"],
        "基础模型在机器人 (Foundation Models for Robotics)": ["foundation model robotics", "large model robotics", "RT-X", "embodied foundation model", "robot foundation model", "embodied pretraining", "foundation model robot"],
        "灵巧操作与人形机器人 (Dexterous Manipulation & Humanoids)": ["dexterous manipulation", "humanoid robot", "bipedal locomotion", "grasping", "manipulation", "hand manipulation", "multi-finger", "fine manipulation"],

        # AI Safety, Alignment & Interpretability 子类别关键词
        "价值对齐与宪法AI (Value Alignment & Constitutional AI)": ["AI alignment", "RLHF", "constitutional AI", "preference optimization", "value alignment", "RLAIF", "human feedback", "DPO", "alignment", "constitutional AI"],
        "机制可解释性 (Mechanistic Interpretability)": ["mechanistic interpretability", "circuit discovery", "superposition", "interpretability", "model interpretability", "explainable AI", "XAI", "feature analysis", "mechanistic interpretability"],
        "幻觉与鲁棒性 (Hallucination & Robustness)": ["hallucination mitigation", "adversarial robustness", "out-of-distribution", "hallucination reduction", "factual accuracy", "factuality", "adversarial attack", "robustness", "adversarial defense"],
        "红队测试与安全评估 (Red Teaming & Safety Evaluation)": ["red teaming", "jailbreak", "AI safety benchmark", "red team", "adversarial testing", "safety testing", "safety evaluation"],
        "隐私与公平性 (Privacy & Fairness)": ["differential privacy AI", "bias mitigation", "poisoning attack", "privacy preservation", "differential privacy", "privacy-preserving", "data privacy", "fairness", "privacy preservation"],

        # Domain-Specific & Personalized AI 子类别关键词
        "个性化大模型 (Personalized LLM)": ["personalized LLM", "personal AI agent", "user adaptation", "personalized language model", "user-specific", "personalization", "custom LLM"],
        "联邦与隐私保护学习 (Federated & Privacy-Preserving Learning)": ["federated learning", "federated", "distributed learning", "privacy-preserving learning", "federated learning"],
        "AI for Science": ["AI for science", "scientific discovery", "scientific AI", "AI research", "AI for science"],
        "医疗健康AI (Medical & Healthcare AI)": ["medical AI", "healthcare AI", "clinical AI", "medical NLP", "health NLP", "medical AI", "healthcare AI"],
        "金融与法律AI (Financial & Legal AI)": ["financial AI", "fintech AI", "financial NLP", "legal AI", "legal NLP", "legal tech", "financial AI", "fintech AI"],
    }
    
    # 计算每个子类别的得分
    subcategory_scores = defaultdict(float)
    for subcategory, keywords in subcategory_keywords.items():
        # 只检查属于当前主类别的子类别
        if subcategory not in subcategories:
            continue
        
        score = 0.0
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # 标题匹配权重更高
            if keyword_lower in title_lower:
                score += 2.0
            # 摘要匹配
            elif keyword_lower in abstract_lower:
                score += 1.0
        
        # 如果没有精确匹配，尝试单词级别的匹配
        if score == 0:
            # 将关键词拆分为单词
            for keyword in keywords:
                keyword_words = keyword.lower().split()
                # 如果关键词是多个单词，检查是否大部分单词出现在标题或摘要中
                if len(keyword_words) > 1:
                    title_word_count = sum(1 for word in keyword_words if word in title_lower)
                    abstract_word_count = sum(1 for word in keyword_words if word in abstract_lower)
                    
                    # 如果大部分关键词单词出现在标题中，给予较高分数（降低阈值从0.7到0.5）
                    if title_word_count >= len(keyword_words) * 0.5:
                        score += 1.5 * (title_word_count / len(keyword_words))
                    # 如果大部分关键词单词出现在摘要中，给予中等分数（降低阈值从0.7到0.5）
                    elif abstract_word_count >= len(keyword_words) * 0.5:
                        score += 0.8 * (abstract_word_count / len(keyword_words))
                else:
                    # 单个单词，检查是否在标题或摘要中
                    if keyword_words[0] in title_lower:
                        score += 1.0
                    elif keyword_words[0] in abstract_lower:
                        score += 0.5
        
        if score > 0:
            subcategory_scores[subcategory] = score
    
    # 如果没有匹配到子类别，返回None
    if not subcategory_scores:
        return (None, 0.0)
    
    # 返回得分最高的子类别
    best_subcategory = max(subcategory_scores.items(), key=lambda x: x[1])
    return best_subcategory


def process_paper(paper, helper, categories_config):
    """处理单篇论文"""
    try:
        # 提取论文信息
        title = paper.title
        authors = ', '.join([author.name for author in paper.authors[:8]])
        abstract = paper.summary
        pdf_url = paper.pdf_url
        published_date = paper.published
        
        # 判断是否是更新的论文
        is_updated = paper.updated > published_date
        
        # 提取代码链接
        code_url = extract_github_link(paper)
        if not code_url:
            code_url = 'None'
        
        # 翻译标题
        title_zh = helper.translate_title(title, abstract)
        
        # 分析论文核心贡献
        contribution_info = helper.analyze_paper_contribution(title, abstract)
        
        # 使用LLM对论文进行分类
        category_results = helper.classify_paper_with_llm(title, abstract)
        if not category_results:
            # 无法分类到预定义类别，略过该论文
            return None
        
        category = category_results[0][0]
        confidence = category_results[0][1]
        subcategory_info = category_results[0][2]
        
        # 提取子类别
        if subcategory_info:
            subcategory = subcategory_info[0]
        else:
            # 如果没有子类别，仍然保留论文（只要有主类别）
            subcategory = ""
        
        # 构建论文字典
        paper_dict = {
            'title': title,
            'title_zh': title_zh,
            'authors': authors,
            'abstract': abstract,
            'pdf_url': pdf_url,
            'github_url': code_url,
            'published_date': published_date,
            'is_updated': is_updated,
            'category': category,
            'subcategory': subcategory,
            'confidence': confidence,
        }
        
        # 添加核心贡献信息
        if contribution_info and "核心贡献" in contribution_info:
            paper_dict["核心贡献"] = contribution_info["核心贡献"]
        
        return paper_dict
        
    except Exception as e:
        print(f"处理论文时出错: {str(e)}")
        traceback.print_exc()
        return None


def get_ai_papers():
    """获取AI/算法论文的主函数"""
    print("=" * 80)
    print("AI/算法论文每日更新系统")
    print("=" * 80)
    
    # 初始化ChatGLM助手
    print("\n🤖 初始化AI助手...")
    helper = ChatGLMHelper()
    
    # 导入关键词配置
    from ai_categories_config import CATEGORY_KEYWORDS
    
    # 创建ArXiv客户端
    print("\n📡 连接ArXiv...")
    client = arxiv.Client(
        page_size=100,  # 每页获取100篇论文
        delay_seconds=5,  # 请求间隔10秒
        num_retries=5    # 失败重试5次
    )

    # 计算目标日期范围（用于过滤）
    target_date = datetime.now() - timedelta(days=QUERY_DAYS_AGO)
    target_date_str = target_date.strftime('%Y-%m-%d')
    
    # 构建查询 - 使用多个类别（不在查询中过滤日期，在代码中过滤）
    query = ' OR '.join([f'cat:{cat}' for cat in ARXIV_CATEGORIES])
    
    search = arxiv.Search(
        query=query,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending  # 确保按时间降序排序
    )

    # 创建线程池
    total_papers = 0
    classified_papers = 0  # 成功分类的论文数
    papers_by_category = defaultdict(list)

    # 使用线程池并行处理论文
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 创建进度条
        print("\n🔍 开始获取论文...")
        results = client.results(search)
        
        # 创建总进度条
        total_pbar = tqdm(
            total=MAX_RESULTS,
            desc="总进度",
            unit="篇",
            position=0,
            leave=True
        )
        
        # 创建批处理进度条
        batch_pbar = tqdm(
            total=0,  # 初始值为0，后面会更新
            desc="当前批次",
            unit="篇",
            position=1,
            leave=True
        )
        
        # 批量处理论文
        batch_size = 10  # 每批处理10篇论文
        papers = []
        futures = []
        
        for result in results:
            if total_papers >= MAX_RESULTS:
                break
            
            # 过滤：只保留目标日期的论文
            paper_date = result.published.strftime('%Y-%m-%d')
            if paper_date != target_date_str:
                continue
            
            papers.append(result)
            total_papers += 1
            total_pbar.update(1)
            
            # 当达到批量大小时，提交处理任务
            if len(papers) >= batch_size:
                batch_pbar.total = len(papers)
                batch_pbar.reset()
                
                # 提交当前批次的所有论文处理任务
                for paper in papers:
                    future = executor.submit(process_paper, paper, helper, CATEGORY_KEYWORDS)
                    futures.append(future)
                
                # 等待当前批次完成
                for future in as_completed(futures):
                    try:
                        paper_dict = future.result()
                        if paper_dict:
                            category = paper_dict['category']
                            papers_by_category[category].append(paper_dict)
                            classified_papers += 1  # 成功分类的论文数+1
                    except Exception as e:
                        print(f"处理论文时出错: {str(e)}")
                        traceback.print_exc()
                
                # 清空批次
                papers = []
                futures = []
                batch_pbar.update(batch_size)
        
        # 处理剩余的论文
        if papers:
            batch_pbar.total = len(papers)
            batch_pbar.reset()
            
            for paper in papers:
                future = executor.submit(process_paper, paper, helper, CATEGORY_KEYWORDS)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    paper_dict = future.result()
                    if paper_dict:
                        category = paper_dict['category']
                        papers_by_category[category].append(paper_dict)
                        classified_papers += 1  # 成功分类的论文数+1
                except Exception as e:
                    print(f"处理论文时出错: {str(e)}")
                    traceback.print_exc()
            
            batch_pbar.update(len(papers))
        
        # 关闭进度条
        total_pbar.close()
        batch_pbar.close()
    
    # 打印统计信息
    # 重新统计实际保存的论文数（有主类别的论文）
    actual_saved_papers = 0
    for category, papers in papers_by_category.items():
        for paper in papers:
            if paper.get('category'):
                actual_saved_papers += 1
    
    print(f"\n📊 统计信息:")
    print(f"获取论文总数: {total_papers} 篇")
    print(f"实际保存论文数: {actual_saved_papers} 篇")
    print(f"未分类论文数: {total_papers - actual_saved_papers} 篇")
    for category, papers in papers_by_category.items():
        if papers:
            print(f"  {category}: {len(papers)} 篇")
    
    # 生成Markdown文件
    print("\n📝 生成Markdown文件...")
    
    # 计算目标日期
    target_date = datetime.now() - timedelta(days=QUERY_DAYS_AGO)
    
    # 创建输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    local_dir = os.path.join(script_dir, '..', 'local')
    
    # 创建年-月目录
    year_month = target_date.strftime('%Y-%m')
    data_year_month = os.path.join(data_dir, year_month)
    local_year_month = os.path.join(local_dir, year_month)
    
    # 确保目录存在
    os.makedirs(data_year_month, exist_ok=True)
    os.makedirs(local_year_month, exist_ok=True)
    
    # 生成文件名
    filename = f"{target_date.strftime('%Y-%m-%d')}.md"
    table_filepath = os.path.join(data_year_month, filename)
    detailed_filepath = os.path.join(local_year_month, filename)

    # 生成标题
    title = f"## [UPDATED!] **{target_date.strftime('%Y-%m-%d')}** (Update Time)\n\n"

    # 保存表格格式的markdown文件到data/年-月目录
    with open(table_filepath, 'w', encoding='utf-8') as f:
        f.write(title)
        f.write(df_to_markdown_table(papers_by_category, target_date))

    # 保存详细格式的markdown文件到local/年-月目录
    with open(detailed_filepath, 'w', encoding='utf-8') as f:
        f.write(title)
        f.write(df_to_markdown_detailed(papers_by_category, target_date))

    print(f"\n表格格式文件已保存到: {table_filepath}")
    print(f"详细格式文件已保存到: {detailed_filepath}")


if __name__ == "__main__":
    # 直接运行查询
    get_ai_papers()
