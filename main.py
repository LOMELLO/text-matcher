#!/usr/bin/env python3
"""Поиск товара по текстовому запросу в терминале.

Использование:
    python main.py                 # обычный запуск
    python main.py --rebuild-db    # пересоздать базу заглушек
    python main.py --demo          # прогнать демо-запросы и выйти
"""

from __future__ import annotations

import argparse
import sys

import database
from matcher import ProductMatcher

DEMO_QUERIES = [
    "хочу купить вешь",
    "нужен новый смартфон Samsung",
    "ищу мышь Logitech со скидкой!!!",
    "где купить наушники sony ???",
    "qqqzzz xxxxyyy непонятно",
]


def print_banner(matcher: ProductMatcher) -> None:
    print("=" * 62)
    print("  ПОИСК ТОВАРА  |  classic ML: TF-IDF триграммы + kNN + LogReg")
    print("=" * 62)
    print(matcher.stats())
    print("Введите запрос словами. Пустой ввод или 'выход' — завершить.\n")


def show(outcome) -> None:
    print(f"\nЗапрос   : {outcome.query}")
    print(f"Очистка  : «{outcome.cleaned}»" if outcome.cleaned else "Очистка  : <пусто>")

    if outcome.found and outcome.best:
        print(f"НАЙДЕНО  : {outcome.best.format()}")
        if outcome.alternatives:
            print("Ещё варианты:")
            for alt in outcome.alternatives:
                print(f"   - {alt.format()}")
    else:
        print("НЕ НАЙДЕНО: " + (outcome.note or "ничего не подошло"))
        if outcome.alternatives:
            print("Возможно, вы имели в виду:")
            for alt in outcome.alternatives:
                print(f"   - {alt.format()}")
    if outcome.note and outcome.found:
        print(f"({outcome.note})")
    print("-" * 62)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Поиск товара по запросу (classic ML)")
    parser.add_argument("--rebuild-db", action="store_true",
                        help="пересоздать products.db с 300 заглушками")
    parser.add_argument("--demo", action="store_true",
                        help="выполнить демо-запросы и завершиться")
    args = parser.parse_args(argv)

    if args.rebuild_db:
        database.create_db(force=True)
        print(f"База пересоздана: {database.DB_PATH}")

    rows = database.load_products()
    print(f"Загружено товаров из БД: {len(rows)}")

    matcher = ProductMatcher().fit(rows)

    if args.demo:
        print_banner(matcher)
        for q in DEMO_QUERIES:
            show(matcher.search(q))
        return 0

    print_banner(matcher)
    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            return 0
        if not line or line.lower() in {"выход", "exit", "quit", "q"}:
            print("Пока!")
            return 0
        show(matcher.search(line))


if __name__ == "__main__":
    sys.exit(main())
