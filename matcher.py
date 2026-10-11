"""Классический ML-матчер наименований товаров.

1. Очистка текста (preprocess.clean_text): мусор, пунктуация, стоп-слова,
   транслитерация кириллицы -> латиница.
2. Признаки: TF-IDF по символьным **триграммам** (ngram_range=(3, 3)).
3. Поиск: kNN (NearestNeighbors, метрика cosine) по корпусу названий.
4. Классификация: логистическая регрессия по 300 классам (товарам),
   обученная на синтетических запросах (выпадение слов, опечатки, мусор).
5. Решение: голосование классификатора и kNN + пороги уверенности.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from preprocess import clean_text

# --- пороги решения -------------------------------------------------------
CLF_CONFIDENT = 0.40   # классификатор сам по себе достаточно уверен
CLF_AGREE = 0.12       # классификатор и kNN согласны -> можно отвечать
KNN_MIN_SIM = 0.25     # минимальный cosine-similarity для kNN-ответа
MIN_QUERY_LEN = 3      # минимальная длина очищенного запроса (для триграмм)

RNG_SEED = 42

#: слова-«мусор», которые добавляются в синтетические запросы при обучении
NOISE_WORDS = [
    "срочно", "новинка", "качество", "доставка", "скидка", "оптом",
    "гарантия", "хороший", "лучший", "быстро", "недорого", "акция",
]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def _typo(word: str, rnd: random.Random) -> str:
    """Вносит опечатку в слово: удаление/замена/перестановка символа."""
    if len(word) < 4:
        return word
    kind = rnd.choice(["del", "swap", "sub"])
    i = rnd.randrange(1, len(word) - 1)
    if kind == "del":
        return word[:i] + word[i + 1:]
    if kind == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + rnd.choice("abcdefghijklmnopqrstuvwxyz") + word[i + 1:]


def build_synthetic_queries(
    cleaned_names: list[str],
    per_product: int = 30,
    seed: int = RNG_SEED,
) -> tuple[list[str], list[int]]:
    """Создаёт синтетические «запросы» для каждого товара.

    Аугментация выполняется поверх очищенных названий:
      * выпадение случайных слов (пользователь пишет не всё название);
      * опечатки;
      * добавление случайного мусорного слова.
    Возвращает (запросы, метки-индексы товаров).
    """
    rnd = random.Random(seed)
    queries: list[str] = []
    labels: list[int] = []

    for idx, name in enumerate(cleaned_names):
        tokens = name.split()
        for _ in range(per_product):
            words = list(tokens)

            # выпадение слов
            while len(words) > 1 and rnd.random() < 0.45:
                del words[rnd.randrange(len(words))]

            # опечатка в одном из слов
            if words and rnd.random() < 0.35:
                j = rnd.randrange(len(words))
                words[j] = _typo(words[j], rnd)

            # мусорное слово в начало/конец
            if rnd.random() < 0.35:
                noise = rnd.choice(NOISE_WORDS)
                words.insert(rnd.randrange(len(words) + 1), noise)

            query = " ".join(words).strip()
            if query:
                queries.append(query)
                labels.append(idx)

    return queries, labels


# ---------------------------------------------------------------------------
# Результаты поиска
# ---------------------------------------------------------------------------
@dataclass
class MatchResult:
    name: str
    category: str
    price: int
    score: float          # уверенность 0..1
    source: str           # откуда пришёл ответ

    def format(self) -> str:
        return f"{self.name}  [{self.category}]  {self.price} руб.  (уверенность {self.score:.0%}, {self.source})"


@dataclass
class SearchOutcome:
    query: str
    cleaned: str
    found: bool
    best: MatchResult | None = None
    alternatives: list[MatchResult] = field(default_factory=list)
    note: str = ""


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------
class ProductMatcher:
    """TF-IDF(триграммы) + kNN + логистическая регрессия."""

    def __init__(
        self,
        clf_confident: float = CLF_CONFIDENT,
        clf_agree: float = CLF_AGREE,
        knn_min_sim: float = KNN_MIN_SIM,
        synthetic_per_product: int = 30,
    ) -> None:
        self.clf_confident = clf_confident
        self.clf_agree = clf_agree
        self.knn_min_sim = knn_min_sim
        self.synthetic_per_product = synthetic_per_product

        self.products: list[tuple[int, str, str, int]] = []
        self.cleaned_names: list[str] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.nn: NearestNeighbors | None = None
        self.clf: LogisticRegression | None = None
        self.train_acc: float = 0.0
        self.test_acc: float = 0.0

    # ------------------------------------------------------------------ fit
    def fit(self, products: list[tuple[int, str, str, int]]) -> "ProductMatcher":
        """Обучает векторизатор, kNN и классификатор на загруженной БД."""
        self.products = products
        self.cleaned_names = [clean_text(name) for _, name, _, _ in products]

        # 1) TF-IDF по символьным триграммам
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 3),   # <-- триграммы
            lowercase=True,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(self.cleaned_names)

        # 2) kNN-поиск (косинусная близость)
        self.nn = NearestNeighbors(n_neighbors=3, metric="cosine")
        self.nn.fit(self.matrix)

        # 3) Синтетические запросы -> обучение логистической регрессии
        queries, labels = build_synthetic_queries(
            self.cleaned_names, self.synthetic_per_product
        )
        X = self.vectorizer.transform(queries)
        y = np.asarray(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=RNG_SEED, stratify=y
        )
        self.clf = LogisticRegression(
            max_iter=2000,
            C=4.0,
            solver="lbfgs",
        )
        self.clf.fit(X_train, y_train)

        self.train_acc = accuracy_score(y_train, self.clf.predict(X_train))
        self.test_acc = accuracy_score(y_test, self.clf.predict(X_test))
        return self

    # --------------------------------------------------------------- search
    def search(self, query: str) -> SearchOutcome:
        """Главный метод: запрос пользователя -> товар из БД."""
        cleaned = clean_text(query)
        if len(cleaned) < MIN_QUERY_LEN:
            return SearchOutcome(
                query=query,
                cleaned=cleaned,
                found=False,
                note="запрос слишком короткий после очистки (нужно >= 3 символов)",
            )

        vec = self.vectorizer.transform([cleaned])

        # --- голос классификатора (логистическая регрессия) ---
        proba = self.clf.predict_proba(vec)[0]
        clf_idx = int(np.argmax(proba))
        clf_p = float(proba[clf_idx])

        # --- голос kNN (косинусная близость) ---
        dists, idxs = self.nn.kneighbors(vec, n_neighbors=3)
        sims = 1.0 - dists[0]
        knn_idx = int(idxs[0][0])
        knn_sim = float(sims[0])

        candidates = {
            i: self._result(i, float(s), "kNN")
            for i, s in zip(idxs[0], sims)
        }

        # --- принятие решения ---
        if clf_idx == knn_idx and clf_p >= self.clf_agree:
            best = self._result(clf_idx, clf_p, "классификатор + kNN")
            alts = [
                r for i, r in candidates.items() if i != clf_idx
            ]
            return SearchOutcome(query, cleaned, True, best, alts)

        if clf_p >= self.clf_confident:
            best = self._result(clf_idx, clf_p, "классификатор")
            alts = [candidates[knn_idx]] if knn_idx != clf_idx else []
            return SearchOutcome(query, cleaned, True, best, alts)

        if knn_sim >= self.knn_min_sim:
            best = candidates[knn_idx]
            alts = [
                r for i, r in candidates.items() if i != knn_idx
            ]
            return SearchOutcome(
                query, cleaned, True, best, alts,
                note="уверенность низкая, показан ближайший по триграммам товар",
            )

        # ничего подходящего
        alts = sorted(candidates.values(), key=lambda r: -r.score)
        return SearchOutcome(
            query, cleaned, False, None, alts,
            note="похоже, такого товара нет в базе",
        )

    # -------------------------------------------------------------- helpers
    def _result(self, idx: int, score: float, source: str) -> MatchResult:
        _, name, category, price = self.products[idx]
        return MatchResult(name, category, price, score, source)

    def stats(self) -> str:
        vocab = len(self.vectorizer.vocabulary_)
        return (
            f"товаров в индексе: {len(self.products)} | "
            f"признаков (триграмм TF-IDF): {vocab} | "
            f"accuracy train: {self.train_acc:.3f} | "
            f"accuracy test: {self.test_acc:.3f}"
        )
