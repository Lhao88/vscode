import jieba
import json

# 计算模型回复答案中有多少token在参考答案中出现过
def calculate_correct_tokens(ref_tokens,can_tokens):
    return [can for can in can_tokens if can in ref_tokens]


def calculate_metrics(references, candidates):
    precisions = []
    recalls = []
    f1_scores = []

    # 多条标准答案列表
    stopwords = ['，', '。', '？', '！', '《', '》', '（', '）', '：', '、', '；', '“', '”']

    for reference, candidate in zip(references, candidates):
        # 使用jieba分词
        ref_tokens = [token for token in jieba.lcut(reference) if token not in stopwords]
        can_tokens = [token for token in jieba.lcut(candidate) if token not in stopwords]

        # 计算正确的token数量（交集）
        correct_tokens = calculate_correct_tokens(ref_tokens, can_tokens)
        num_correct = len(correct_tokens)

        # 计算精确率Precision
        precision = num_correct / len(can_tokens) if can_tokens else 0
        precisions.append(precision)

        # 计算召回率Recall
        recall = num_correct / len(ref_tokens) if ref_tokens else 0
        recalls.append(recall)

        # 计算F1值
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    # 计算平均Precision、Recall、F1
    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    return avg_precision, avg_recall, avg_f1

# 导入问题列表
def load_data(file_path):
    with open(file_path,'r',encoding='UTF-8') as json_file:
        data = json.load(json_file)
    return data



if __name__ == '__main__':

    # results = load_data('response/power_qa_rag_results_qwen_lora.json')
    # results = load_data('response/power_qa_results_qwen.json')  # 原始模型回复
    # results = load_data('response/power_qa_rag_results_qwen.json')  # 原始模型+RAG 回复
    results = load_data('response/power_qa_results_qwen_lora.json')  # 微调模型回复
    # 多条标准答案列表
    references = []
    # 多条模型回答列表
    candidates = []
    for r in results:
        references.append(r['reference'])
        candidates.append(r['response'])
    print(len(references))
    print(len(candidates))

    avg_precision, avg_recall, avg_f1 = calculate_metrics(references, candidates)
    print(f"Average Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average F1: {avg_f1}")