import unittest
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from score_fn import score_fn, extract_entities


class TestScoreFunction(unittest.TestCase):
    """Проверка вычисления precision, recall и f1 для разных сценариев"""
    def test_score_cases(self):
        cases = [
            # perfect match
            ("Москва|LOC,Путин|PER,Кремль|ORG", "Москва|LOC,Путин|PER,Кремль|ORG", 1.0, 1.0, 1.0),
            # half match
            ("Москва|LOC,Путин|PER,Кремль|ORG,Россия|LOC", "Москва|LOC,Путин|PER,Белый дом|ORG,США|LOC", 0.5, 0.5, 0.5),
            # partial match
            ("Москва|LOC,Владимир Путин|PER", "Москва|LOC,Путин|PER,Иванов|PER", 1/3, 0.5, 0.4),
            # precision > recall
            ("Москва|LOC,Путин|PER,Кремль|ORG,Россия|LOC,Петров|PER", "Москва|LOC,Путин|PER", 1.0, 0.4, 2*(1.0*0.4)/(1.0+0.4)),
            # no match
            ("Москва|LOC,Путин|PER", "Вашингтон|LOC,Байден|PER", 0.0, 0.0, 0.0),
            # overprediction
            ("Москва|LOC", "Москва|LOC,Путин|PER,Кремль|ORG,Россия|LOC,Петров|PER", 0.2, 1.0, 2*(0.2*1.0)/(0.2+1.0)),
            # underprediction
            ("Москва|LOC,Путин|PER,Кремль|ORG", "", 0.0, 0.0, 0.0),
            # empty gold
            ("", "Москва|LOC,Путин|PER", 0.0, 0.0, 0.0),
            # both empty
            ("", "", 0.0, 0.0, 0.0),
            # malformed string ignored
            ("Москва|LOC,ПутинБезТипа,Кремль|ORG", "Москва|LOC", 1.0, 0.5, 2*(1.0*0.5)/(1.0+0.5)),
        ]
        for gold, pred, exp_p, exp_r, exp_f in cases:
            with self.subTest(gold=gold, pred=pred):
                p = score_fn(gold, pred, 'precision')
                r = score_fn(gold, pred, 'recall')
                f = score_fn(gold, pred, 'f1')
                self.assertAlmostEqual(p, exp_p, places=3)
                self.assertAlmostEqual(r, exp_r, places=3)
                self.assertAlmostEqual(f, exp_f, places=3)


class TestParseEntities(unittest.TestCase):
    """Проверка разбора строки сущностей"""
    def test_empty_string(self):
        self.assertEqual(extract_entities(''), set())

    def test_single_entity(self):
        self.assertEqual(extract_entities('Entity|TYPE'), {('Entity', 'TYPE')})

    def test_multiple_entities(self):
        s = 'A|X,B|Y,C|Z'
        self.assertEqual(extract_entities(s), {('A', 'X'), ('B', 'Y'), ('C', 'Z')})

    def test_ignore_malformed(self):
        self.assertEqual(extract_entities('no_separator'), set())
        self.assertEqual(extract_entities('A|'), set())
        self.assertEqual(extract_entities('A|X,,B|Y'), {('A', 'X'), ('B', 'Y')})

if __name__ == '__main__':
    unittest.main()
