import pandas as pd
import numpy as np

def extract_entities(s):
    """
    Разбирает строку вида "text1|TYPE1,text2|TYPE2,..." в множество кортежей.
    Args:
        s: Строка с перечислением сущностей через запятую. 
           Каждый элемент — пара текст и тип через '|'.

    Returns:
        Множество кортежей (normalized_text, type).
    """
    if not s:
        return set()
    entities = set()
    for item in s.split(','):
        parts = item.strip().split('|', 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            entities.add((parts[0], parts[1]))
    return entities

def score_fn(gold, pred, metric='f1'):
    gold_set = extract_entities(gold)
    pred_set = extract_entities(pred)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    if metric == 'precision':
        return precision
    if metric == 'recall':
        return recall
    if metric == 'f1':
        return 2*precision*recall/(precision+recall) if precision+recall else 0.0
    raise ValueError(f"Неизвестная метрика: '{metric}'. Доступны: 'precision', 'recall', 'f1'" )

def vectorized_score_fn(golds, preds, metric='f1'):
    vec = np.vectorize(lambda g, p: score_fn(g, p, metric), otypes=[float])
    return pd.Series(vec(golds.values, preds.values))