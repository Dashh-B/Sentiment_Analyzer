# -*- coding: utf-8 -*-
"""
analyze_sentiments.py — Анализ тональности отзывов из JSON.

Пайплайн:
  1. Парсит JSON из указанных файлов или /content
  2. Анализирует тональность каскадной моделью
  3. Сохраняет результат в CSV

Запуск:
  python analyze_sentiments.py --input content/resident_complex_reviews_report.json

  # Несколько файлов:
  python analyze_sentiments.py --input content/resident_complex_reviews_report.json content/school_reviews_report.json

  # Указать выходной файл:
  python analyze_sentiments.py --input content/resident_complex_reviews_report.json --output [результаты].csv
"""

import os
import sys
import time
import argparse
from pathlib import Path

import warnings
import numpy as np
import pandas as pd

from json_parser import load_all_json

# Подавляем служебный вывод huggingface/sentence_transformers при импорте
import io as _io
_stderr = sys.stderr
sys.stderr = _io.StringIO()
try:
    from sentence_transformers import SentenceTransformer as _ST
finally:
    sys.stderr = _stderr

warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning)

# Подавляем информационный вывод библиотек при загрузке моделей
import logging
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

# ──────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────

CONTENT_DIR           = Path('/content')
BINARY_MODEL_PATH     = 'models/best_binary_model.pkl'
MULTICLASS_MODEL_PATH = 'models/best_multiclass_model.pkl'
MODEL_NAME            = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
THRESHOLD             = 0.95
MIN_DIFF              = 0.20
LABEL_MAP             = {0: 'Негатив', 1: 'Позитив', 2: 'Смешанный'}

# ──────────────────────────────────────────────
# Утилиты вывода
# ──────────────────────────────────────────────

def color(text, code):
    codes = {'green': '32', 'red': '31', 'yellow': '33', 'cyan': '36', 'bold': '1'}
    return f"\033[{codes.get(code, '0')}m{text}\033[0m"

def print_step(msg): print(f'\n{color("="*54, "cyan")}\n  {msg}\n{color("="*54, "cyan")}')
def print_ok(msg):   print(f'  {color("v", "green")} {msg}')
def print_info(msg): print(f'  . {msg}')

# ──────────────────────────────────────────────
# Загрузка моделей
# ──────────────────────────────────────────────

def load_models():
    try:
        import joblib
    except ImportError as e:
        sys.exit(color(f'Ошибка импорта: {e}', 'red'))

    for path in [BINARY_MODEL_PATH, MULTICLASS_MODEL_PATH]:
        if not os.path.exists(path):
            sys.exit(color(
                f'Файл модели не найден: {path}\n'
                f'  Сначала запустите: python train_models.py', 'red'
            ))

    binary_model     = joblib.load(BINARY_MODEL_PATH)
    multiclass_model = joblib.load(MULTICLASS_MODEL_PATH)

    st_model = _ST(MODEL_NAME)

    print_ok('Модели загружены')
    return binary_model, multiclass_model, st_model


# ──────────────────────────────────────────────
# Каскадный классификатор
# ──────────────────────────────────────────────

def predict_cascade(emb, binary_model, multiclass_model):
    single = [emb]
    try:
        pb    = binary_model.predict_proba(single)[0]
        max_p = float(np.max(pb))
        dif_p = float(abs(pb[0] - pb[1]))
        if max_p >= THRESHOLD and dif_p >= MIN_DIFF:
            pred = int(binary_model.predict(single)[0])
            return LABEL_MAP[pred], max_p, 'binary'
    except AttributeError:
        pred = int(binary_model.predict(single)[0])
        return LABEL_MAP[pred], 1.0, 'binary'

    try:
        pm       = multiclass_model.predict_proba(single)[0]
        pred_idx = int(np.argmax(pm))
        conf     = float(pm[pred_idx])
    except Exception:
        pred_idx = int(multiclass_model.predict(single)[0])
        conf     = 1.0

    return LABEL_MAP[pred_idx], conf, 'multiclass'


def analyze_sentiment(df, binary_model, multiclass_model, st_model):
    print_info(f'Анализируем {len(df)} отзывов...')
    t0   = time.time()
    embs = st_model.encode(
        df['review_text'].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    sentiments, confidences, sources = [], [], []
    for emb in embs:
        label, conf, source = predict_cascade(emb, binary_model, multiclass_model)
        sentiments.append(label)
        confidences.append(round(conf, 4))
        sources.append(source)

    df = df.copy()
    df['sentiment']    = sentiments
    df['confidence']   = confidences
    df['model_source'] = sources

    elapsed = time.time() - t0
    print_ok(f'Анализ завершен за {elapsed:.0f} с ({elapsed/len(df)*1000:.0f} мс/отзыв)')
    return df


# ──────────────────────────────────────────────
# Итоговый отчет
# ──────────────────────────────────────────────

def print_summary(df):
    from collections import Counter
    print()
    print(color('╔══════════════════════════════════════════════════════╗', 'cyan'))
    print(color('║                    Итоговый отчет                    ║', 'bold'))
    print(color('╚══════════════════════════════════════════════════════╝', 'cyan'))

    total        = len(df)
    counts       = Counter(df['sentiment'])
    binary_count = (df['model_source'] == 'binary').sum()
    avg_conf     = df['confidence'].mean()

    print(f'\n  Всего отзывов:  {total}')
    for label in ['Позитив', 'Негатив', 'Смешанный']:
        cnt = counts.get(label, 0)
        pct = cnt / total * 100
        bar = '#' * int(pct / 4)
        print(f'  {label:<12} {cnt:>5} ({pct:5.1f}%)  {bar}')

    print(f'\n  Средняя уверенность: {avg_conf:.1%}')
    print(f'  Бинарная модель:     {binary_count} ({binary_count/total:.1%})')

    if 'group_name' in df.columns and df['group_name'].nunique() > 1:
        print(f'\n  По типу объекта:')
        for group in df['group_name'].unique():
            sub = df[df['group_name'] == group]
            pos = (sub['sentiment'] == 'Позитив').sum()
            neg = (sub['sentiment'] == 'Негатив').sum()
            mix = (sub['sentiment'] == 'Смешанный').sum()
            print(f'    {group:<12} всего={len(sub)}  pos={pos}  neg={neg}  mix={mix}')

    if 'name' in df.columns:
        neg_by_obj = (
            df[df['sentiment'] == 'Негатив']
            .groupby('name').size()
            .sort_values(ascending=False).head(5)
        )
        if len(neg_by_obj) > 0:
            print(f'\n  Топ-5 объектов по числу негативных отзывов:')
            for name, cnt in neg_by_obj.items():
                short = name[:45] + '...' if len(name) > 45 else name
                print(f'    {color("-", "red")} {short}: {cnt}')
    print()


# ──────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Анализ тональности отзывов из JSON',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--input', '-i', nargs='+', default=None,
        help=f'JSON-файлы для анализа.\nПо умолчанию: все .json из {CONTENT_DIR}'
    )
    parser.add_argument(
        '--output', '-o', default='data/analyzed_reviews.csv',
        help='Путь к выходному CSV (по умолчанию: data/analyzed_reviews.csv)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print()
    print(color('╔══════════════════════════════════════════════════════╗', 'cyan'))
    print(color('║             Анализ тональности отзывов               ║', 'bold'))
    print(color('╚══════════════════════════════════════════════════════╝', 'cyan'))

    # ── Шаг 1: Парсинг ──
    if args.input:
        filepaths = [Path(p) for p in args.input]
    else:
        if not CONTENT_DIR.exists():
            sys.exit(color(
                f'Папка {CONTENT_DIR} не найдена. '
                f'Укажите файлы: --input файл.json', 'red'
            ))
        filepaths = sorted(CONTENT_DIR.glob('*.json'))
        if not filepaths:
            sys.exit(color(f'В папке {CONTENT_DIR} нет .json файлов.', 'red'))

    print_step('Парсинг JSON-файлов')
    print_info(f'Файлов: {len(filepaths)}')
    for p in filepaths:
        print_info(f'  {p}')

    df = load_all_json(filepaths, with_sentiment=False)
    if len(df) == 0:
        sys.exit(color('Не удалось извлечь ни одного отзыва.', 'red'))
    print_ok(f'Всего отзывов: {len(df)}')

    # ── Шаг 2: Загрузка моделей ──
    print_step('Загрузка моделей')
    binary_model, multiclass_model, st_model = load_models()

    # ── Шаг 3: Анализ ──
    print_step('Анализ тональности')
    df = analyze_sentiment(df, binary_model, multiclass_model, st_model)

    # ── Шаг 4: Сохранение ──
    print_step('Сохранение результатов')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print_ok(f'Результат: {args.output}')
    print_info(f'Колонки: {", ".join(df.columns.tolist())}')

    # ── Шаг 5: Отчет ──
    print_summary(df)


if __name__ == '__main__':
    main()