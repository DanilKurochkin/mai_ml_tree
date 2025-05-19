import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator


def find_best_split(feature_vector: np.ndarray, target_vector: np.ndarray):
    """
    Находит оптимальный порог для разбиения вектора признака по критерию Джини.

    Parameters
    ----------
    feature_vector : np.ndarray
        Вектор вещественнозначных значений признака.
    target_vector : np.ndarray
        Вектор классов объектов (0 или 1), длина `feature_vector` равна длине `target_vector`.

    Returns
    -------
    thresholds : np.ndarray
        Отсортированный по возрастанию вектор возможных порогов.
    gini_scores : np.ndarray
        Вектор значений критерия Джини для каждого порога.
    best_threshold : float
        Оптимальный порог разбиения.
    best_gini : float
        Минимальное значение критерия Джини.
    """
    # Сортировка по значениям признака
    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]

    # Индексы, где признак изменяется
    split_indices = np.where(sorted_features[:-1] != sorted_features[1:])[0]
    if len(split_indices) == 0:
        return np.array([]), np.array([]), None, np.inf

    # Возможные пороги — среднее между соседними различающимися значениями
    thresholds = (sorted_features[split_indices] + sorted_features[split_indices + 1]) / 2

    # Кумулятивная сумма целевых значений
    cumulative_positive = np.cumsum(sorted_targets)
    total_samples = len(sorted_targets)
    total_positives = cumulative_positive[-1]

    left_sizes = split_indices + 1
    right_sizes = total_samples - left_sizes

    left_positives = cumulative_positive[split_indices]
    right_positives = total_positives - left_positives

    # Пропорции классов в левой и правой части
    left_positive_ratio = left_positives / left_sizes
    left_negative_ratio = 1 - left_positive_ratio
    left_gini = 1 - left_positive_ratio**2 - left_negative_ratio**2

    right_positive_ratio = right_positives / right_sizes
    right_negative_ratio = 1 - right_positive_ratio
    right_gini = 1 - right_positive_ratio**2 - right_negative_ratio**2

    # Суммарный Gini для каждого порога
    gini_scores = (left_sizes / total_samples) * left_gini + (right_sizes / total_samples) * right_gini

    # Выбор наилучшего порога
    best_index = np.argmin(gini_scores)
    best_threshold = thresholds[best_index]
    best_gini = gini_scores[best_index]

    return thresholds, gini_scores, best_threshold, best_gini



class DecisionTree:
    def __init__(
        self,
        feature_types,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    ):
        if any(ft not in {"real", "categorical"} for ft in feature_types):
            raise ValueError("Неизвестный тип признака")

        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._tree = {}


    def _fit_node(self, X_subset, y_subset, node, depth=0):
        # Если все метки одинаковы — терминальный узел
        if np.all(y_subset == y_subset[0]):
            node["type"] = "terminal"
            node["class"] = int(y_subset[0])
            return

        # Проверка условий остановки
        if (self._max_depth is not None and depth >= self._max_depth) or len(y_subset) < self._min_samples_split:
            most_common_class = Counter(y_subset).most_common(1)[0][0]
            node["type"] = "terminal"
            node["class"] = int(most_common_class)
            return

        best_feature = None
        best_threshold = None
        best_gini = None
        best_split_mask = None
        best_category_map = None

        num_features = X_subset.shape[1]

        for feature_index in range(num_features):
            feature_type = self._feature_types[feature_index]
            column_values = X_subset[:, feature_index]

            if feature_type == "real":
                feature_vector = column_values.astype(float)
                category_map = None
            else:
                # Упорядочим категории по вероятности положительного класса
                category_counts = Counter(column_values)
                positive_counts = Counter(column_values[y_subset == 1])
                click_ratios = {cat: positive_counts.get(cat, 0) / category_counts[cat] for cat in category_counts}
                sorted_categories = sorted(click_ratios, key=click_ratios.get)
                category_map = {cat: idx for idx, cat in enumerate(sorted_categories)}
                feature_vector = np.array([category_map.get(val, -1) for val in column_values])

            # Пропускаем константные
            if len(np.unique(feature_vector)) <= 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, y_subset)

            if threshold is None or gini is None:
                continue

            if best_gini is None or gini < best_gini:
                best_feature = feature_index
                best_threshold = threshold
                best_gini = gini
                best_split_mask = feature_vector < threshold
                best_category_map = category_map

        # Если сплит невозможен или нарушены ограничения по количеству объектов в листьях
        if (
            best_feature is None or
            best_split_mask.sum() < self._min_samples_leaf or
            (~best_split_mask).sum() < self._min_samples_leaf
        ):
            majority_class = Counter(y_subset).most_common(1)[0][0]
            node["type"] = "terminal"
            node["class"] = int(majority_class)
            return

        # Создаём внутренний узел
        node["type"] = "nonterminal"
        node["feature_split"] = best_feature
        feature_type = self._feature_types[best_feature]

        if feature_type == "real":
            node["threshold"] = float(best_threshold)
        else:
            # Категории, идущие в левое поддерево
            node["categories_split"] = [
                category for category, index in best_category_map.items() if index < best_threshold
            ]

        # Рекурсивно обучаем детей
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(X_subset[best_split_mask], y_subset[best_split_mask], node["left_child"], depth + 1)
        self._fit_node(X_subset[~best_split_mask], y_subset[~best_split_mask], node["right_child"], depth + 1)


    def _predict_node(self, x, node):
        """
        Рекурсивное предсказание класса для одного объекта по узлу дерева решений.

        Если узел терминальный, возвращается предсказанный класс.
        Если узел не терминальный, выборка передается в соответствующее поддерево для дальнейшего предсказания.

        Parameters
        ----------
        x : np.ndarray
            Вектор признаков одного объекта.
        node : dict
            Узел дерева решений.

        Returns
        -------
        int
            Предсказанный класс объекта.
        """
        if node["type"] == "terminal":
            return node["class"]

        feature_idx = node["feature_split"]
        feature_type = self._feature_types[feature_idx]

        if feature_type == "real":
            threshold = node.get("threshold")
            return self._predict_node(
                x, node["left_child"] if float(x[feature_idx]) < threshold else node["right_child"]
            )
        else:
            category = x[feature_idx]
            left_categories = node.get("categories_split", [])
            return self._predict_node(
                x, node["left_child"] if category in left_categories else node["right_child"]
            )

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, depth=0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
