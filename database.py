"""База данных товаров на SQLite.

При первом запуске создаётся файл ``products.db`` с 300 наименованиями
товаров-заглушек. Среди них — «Super vesh 3000», который и должен
находить пользователь, введя «хочу купить вешь».

Часть названий в БД намеренно «грязная» (лишние пробелы, КАПС,
пунктуация) — чтобы было видно, как работает очистка текста.
"""

from __future__ import annotations

import random
import re
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().with_name("products.db")

#: Товар, который обязан находиться по запросу «хочу купить вешь».
TARGET_NAME = "Super vesh 3000"

TOTAL_PRODUCTS = 300  # 300 заглушек, как в задании

# Категории и «модели» для генерации заглушек
CATEGORIES: dict[str, list[str]] = {
    "Смартфон": [
        "Samsung Galaxy A55", "Apple iPhone 13", "Xiaomi Redmi Note 13",
        "Google Pixel 8", "OnePlus 12", "Realme 12 Pro", "Honor X9b",
        "Nothing Phone 2",
    ],
    "Ноутбук": [
        "ASUS Vivobook X1504", "Lenovo IdeaPad 3", "Acer Aspire 5",
        "Apple MacBook Air M3", "HP Pavilion 15", "MSI Thin 15",
        "Dell Inspiron 14", "Huawei MateBook D14",
    ],
    "Наушники": [
        "Sony WH-1000XM5", "JBL Tune 770", "Apple AirPods Pro 2",
        "Xiaomi Buds 4", "Sennheiser HD 450BT", "Marshall Major IV",
        "Soundcore Q30", "Bose QuietComfort 45",
    ],
    "Клавиатура": [
        "Logitech G Pro X", "Keychron K2", "Razer BlackWidow V4",
        "HyperX Alloy Origins", "Ducky One 3", "A4Tech Bloody B820",
        "Defender Marathon", "MSI Vigor GK30",
    ],
    "Мышь": [
        "Logitech G502 Hero", "Razer DeathAdder V3", "SteelSeries Rival 3",
        "Xiaomi Mi Wireless", "A4Tech Bloody W70", "HyperX Pulsefire Haste",
        "ASUS TUF M3", "Delux M800",
    ],
    "Монитор": [
        "Samsung Odyssey G5", "LG UltraGear 27GR", "Dell U2422H",
        "AOC 24G2U", "ASUS TUF VG279", "Philips 242E1GSB",
        "ViewSonic VX2405", "BenQ EX2780Q",
    ],
    "Телевизор": [
        "LG OLED55C3", "Samsung QE55Q60C", "Sony KD-55X80L",
        "Xiaomi TV A Pro 55", "TCL 55C645", "Haier LE55U5000",
        "Panasonic TX-55MX800", "BBK 55LEM-1002",
    ],
    "Пылесос": [
        "Roborock Q7", "Xiaomi Mi Robot Vacuum", "Dyson V12 Detect",
        "Samsung Jet 70", "Aicle T8", "Deerma DX700", "Kitfort KT-511",
        "Polaris PVC 0910",
    ],
    "Кофемашина": [
        "DeLonghi Magnifica", "Philips EP2231", "Krups EA89",
        "Jura E8", "Bosch Tassimo", "Redmond RCM-1503", "Polaris PCM 1576",
        "Moulinex Barista",
    ],
    "Умные часы": [
        "Apple Watch SE 2", "Samsung Galaxy Watch 6", "Xiaomi Watch S3",
        "Huawei Watch GT 4", "Amazfit GTR 4", "Garmin Venu 3",
        "Honor Watch 4", "Hoco W30",
    ],
    "Видеокарта": [
        "NVIDIA RTX 4070", "AMD RX 7800 XT", "NVIDIA RTX 4060",
        "AMD RX 7600", "Gigabyte Gaming OC", "Palit JetStream",
        "MSI Ventus 3X", "ASUS Dual Edition",
    ],
    "SSD накопитель": [
        "Samsung 980 PRO", "Kingston NV2", "WD Black SN850X",
        "Crucial P3 Plus", "ADATA XPG Gammix", "T-Force Cardea",
        "Silicon Power P34", "Goodram CX400",
    ],
    "Колонка": [
        "JBL Charge 5", "Xiaomi Mi Speaker 3", "Marshall Stanmore",
        "Yandex Station 2", "Apple HomePod mini", "SRS-XB100",
        "Dexp RSP-200", "M-Audio Block Rocka",
    ],
    "Роутер": [
        "TP-Link Archer C6", "Xiaomi Mi Router 4A", "ASUS RT-AX55",
        "Keenetic Viva", "Tenda AC23", "Zyxel Keenetic", "MikroTik hAP",
        "D-Link DIR-825",
    ],
    "Фен": [
        "Dyson Supersonic", "Rowenta CV5820", "Philips BHD007",
        "Brauns 5795", "REDMOND RHD-2200", "Kitfort KT-761",
        "Polaris PHD 1460", "Hair Rush HR-2200",
    ],
    "Электросамокат": [
        "Xiaomi Scooter Pro 2", "Kugoo M4 Pro", "Ninebot Max G30",
        "Hiper S5", "Eljet Route", "Dji Onewheel", "Omnio Swift",
        "Eco-Ride E2",
    ],
}

#: Дополнительные «хвосты» названий (цвет, объём, версия)
ATTRIBUTES = [
    "черный", "белый", "серый", "128GB", "256GB", "512GB", "1TB",
    "Pro", "Max", "Lite", "2024", "5G", "Wi-Fi", "RGB", "Серебро",
    "мини", "V2", "PLUS",
]


def _dirty(name: str, rnd: random.Random) -> str:
    """Намеренно «замылять» название, как в реальных выгрузках из 1С."""
    roll = rnd.random()
    if roll < 0.14:
        return name.upper()
    if roll < 0.26:
        return f"   {name}   "
    if roll < 0.34:
        return f"{name} !!!"
    if roll < 0.40:
        return f"{name} – NEW"
    if roll < 0.46:
        return re.sub(r" {2,}", "   ", name)  # двойные пробелы
    if roll < 0.52:
        return f"{name} /// актуально?"
    return name


def _generate_products(count: int) -> list[tuple[str, str, int]]:
    """Генерирует ``count`` уникальных (название, категория, цена)."""
    rnd = random.Random(42)
    cats = list(CATEGORIES)
    used = {TARGET_NAME.lower()}
    result: list[tuple[str, str, int]] = [
        (TARGET_NAME, "Коллекционный товар", 9990)
    ]

    while len(result) < count:
        idx = len(result)
        category = cats[idx % len(cats)]
        model = rnd.choice(CATEGORIES[category])
        attr = rnd.choice(ATTRIBUTES)
        name = f"{category} {model} {attr}"
        if name.lower() in used:  # на всякий случай — артикул
            name = f"{name} MKT-{idx:03d}"
        if name.lower() in used:
            continue
        used.add(name.lower())
        name = _dirty(name, rnd)
        price = rnd.randint(499, 189_999)
        result.append((name, category, price))

    return result


def create_db(path: Path = DB_PATH, force: bool = False) -> bool:
    """Создаёт БД с заглушками. Возвращает True, если БД была создана."""
    if path.exists() and not force:
        return False
    if path.exists() and force:
        path.unlink()

    products = _generate_products(TOTAL_PRODUCTS)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE products (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                name     TEXT NOT NULL UNIQUE,
                category TEXT NOT NULL,
                price    INTEGER NOT NULL
            )
            """
        )
        conn.executemany(
            "INSERT INTO products (name, category, price) VALUES (?, ?, ?)",
            products,
        )
        conn.commit()
    finally:
        conn.close()
    return True


def load_products(path: Path = DB_PATH) -> list[tuple[int, str, str, int]]:
    """Загружает товары из БД: [(id, name, category, price), ...]."""
    if not path.exists():
        create_db(path)
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            "SELECT id, name, category, price FROM products ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return rows


if __name__ == "__main__":
    created = create_db(force=True)
    rows = load_products()
    print(f"Создано: {created}, товаров в БД: {len(rows)}")
    for row in rows[:5]:
        print(row)
