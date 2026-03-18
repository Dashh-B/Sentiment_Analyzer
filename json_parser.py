# -*- coding: utf-8 -*-
"""
json_parser.py — Парсинг JSON-файлов с отзывами.

Поддерживаемые форматы:
  - Жилые комплексы: { "resident_complexes": [...], "timestamp": "..." }
  - Школы:           { "schools": [...],            "timestamp": "..." }
  - Оба формата в одном файле

Используется как модуль из train_models.py и analyze_sentiments.py:
  from json_parser import load_all_json
"""

import os
import re
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


# ──────────────────────────────────────────────
# Очистка текста
# ──────────────────────────────────────────────

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r"[^а-яА-ЯёЁa-zA-Z0-9.,:;!?\"() -]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ──────────────────────────────────────────────
# Парсинг дат
# ──────────────────────────────────────────────

def relative_date_to_months_ago(text):
    """Конвертирует относительную дату ('3 месяца назад') в количество месяцев."""
    if not isinstance(text, str):
        return None
    text = text.strip().lower()

    m = re.search(r'(\d+)\s+(?:год|лет|года)\s+назад', text)
    if m: return int(m.group(1)) * 12

    m = re.search(r'(\d+)\s+(?:месяц[а-я]*|мес[.]?)\s+назад', text)
    if m: return int(m.group(1))

    if re.search(r'полгода',        text): return 6
    if re.search(r'полтора\s+года', text): return 18
    if re.search(r'год\s+назад',    text): return 12
    if re.search(r'несколько.{0,10}недел', text): return 1
    if re.search(r'несколько.{0,10}дней',  text): return 0
    if re.search(r'вчера|сегодня',  text): return 0

    return None


def parse_date(raw_date, base_dt):
    """Возвращает datetime из строки даты (абсолютной или относительной)."""
    if not isinstance(raw_date, str) or not raw_date.strip():
        return None

    for fmt in ('%d.%m.%Y', '%Y-%m-%d', '%d/%m/%Y', '%Y.%m.%d'):
        try:
            return datetime.strptime(raw_date.strip(), fmt)
        except ValueError:
            continue

    months_ago = relative_date_to_months_ago(raw_date)
    if months_ago is None:
        return None

    try:
        total   = base_dt.month - months_ago
        years   = base_dt.year + (total // 12)
        months  = total % 12 or 12
        max_day = (datetime(years, months % 12 + 1, 1) - timedelta(days=1)).day
        return datetime(years, months, min(base_dt.day, max_day))
    except Exception:
        return None


# ──────────────────────────────────────────────
# Маппинг рейтинга
# ──────────────────────────────────────────────

def rating_to_sentiment(rating):
    """1-2 -> Негатив, 3 -> Смешанный, 4-5 -> Позитив."""
    try:
        r = float(rating)
        if r <= 2:   return 'Негатив'
        elif r == 3: return 'Смешанный'
        else:        return 'Позитив'
    except (TypeError, ValueError):
        return None


# ──────────────────────────────────────────────
# Парсинг файлов
# ──────────────────────────────────────────────

def parse_json_file(filepath, with_sentiment=False):
    """
    Парсит один JSON-файл с отзывами.

    Параметры:
        filepath       — путь к файлу
        with_sentiment — добавлять колонку sentiment из рейтинга
                         (True для обучения, False для анализа)

    Возвращает DataFrame с колонками:
        object_id, name, address, group_name, user_name,
        date, review_rating, review_text, [sentiment]
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f'  [!] Файл не найден: {filepath}')
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        print(f'  [!] Ошибка JSON в {os.path.basename(filepath)}: {e}')
        return pd.DataFrame()

    base_dt = datetime.now()
    if 'timestamp' in data:
        try:
            ts = str(data['timestamp']).replace('Z', '+00:00')
            base_dt = datetime.fromisoformat(ts).replace(tzinfo=None)
        except Exception:
            pass

    formats = []
    if 'resident_complexes' in data: formats.append(('resident_complexes', 'residents'))
    if 'schools'            in data: formats.append(('schools',            'schools'))

    if not formats:
        print(f'  [!] Неизвестный формат в {os.path.basename(filepath)}: '
              f'нет ключей "resident_complexes" или "schools"')
        return pd.DataFrame()

    rows = []
    for key, group_name in formats:
        for obj in data.get(key, []):
            obj_id  = obj.get('id') or obj.get('school_id', '')
            name    = obj.get('name', '')
            address = obj.get('address', '')

            for review in obj.get('reviews', []):
                text   = clean_text(review.get('text', ''))
                rating = review.get('rating')

                if not text:
                    continue

                row = {
                    'object_id':     obj_id,
                    'name':          name,
                    'address':       address,
                    'group_name':    group_name,
                    'user_name':     review.get('user_name', ''),
                    'date':          parse_date(review.get('date', ''), base_dt),
                    'review_rating': rating,
                    'review_text':   text,
                }

                if with_sentiment:
                    label = rating_to_sentiment(rating)
                    if label is None:
                        continue  # пропускаем отзывы без рейтинга при обучении
                    row['sentiment'] = label

                rows.append(row)

    df = pd.DataFrame(rows)
    print(f'  [ok] {os.path.basename(filepath)}: {len(df)} отзывов'
          + (f'  {df["sentiment"].value_counts().to_dict()}' if with_sentiment and len(df) > 0 else ''))
    return df


def load_all_json(filepaths, with_sentiment=False):
    """
    Парсит список JSON-файлов и объединяет результат.

    Параметры:
        filepaths      — список путей к файлам
        with_sentiment — передается в parse_json_file

    Возвращает объединенный DataFrame без дублей.
    """
    frames = [parse_json_file(p, with_sentiment=with_sentiment) for p in filepaths]
    frames = [f for f in frames if len(f) > 0]

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=['name', 'user_name', 'review_text'])
    return df
