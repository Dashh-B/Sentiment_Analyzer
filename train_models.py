# -*- coding: utf-8 -*-
"""
train_models.py — Обучение каскадных моделей тональности отзывов из JSON.

Пайплайн:
  1. Парсит JSON из указанных файлов или /content
  2. Балансирует датасет, сохраняет в data/dataset.csv
  3. Строит sentence-transformer эмбеддинги
  4. Обучает бинарную и многоклассовую LightGBM-модели
  5. Сохраняет модели в models/

Запуск:
  python train_models.py --input content/resident_complex_reviews_report.json content/school_reviews_report.json

  # Только сформировать датасет без обучения:
  python train_models.py --input content/resident_complex_reviews_report.json --dataset-only
"""

import sys
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier
from sentence_transformers import SentenceTransformer

from json_parser import load_all_json

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# Пути
# ──────────────────────────────────────────────

CONTENT_DIR           = Path('/content')
DATA_DIR              = Path('/data')
MODELS_DIR            = Path('/models')
DATASET_PATH          = DATA_DIR / 'dataset.csv'
BINARY_MODEL_PATH     = MODELS_DIR / 'best_binary_model.pkl'
MULTICLASS_MODEL_PATH = MODELS_DIR / 'best_multiclass_model.pkl'

# ──────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────

MODEL_NAME   = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
VALID_LABELS = ['Негатив', 'Позитив', 'Смешанный']
THRESHOLD    = 0.95
MIN_DIFF     = 0.20
LGBM_PARAMS  = dict(n_estimators=300, num_leaves=31, class_weight='balanced', verbosity=-1)

# ──────────────────────────────────────────────
# Утилиты вывода
# ──────────────────────────────────────────────

def color(text, code):
    codes = {'green': '32', 'red': '31', 'yellow': '33', 'cyan': '36', 'bold': '1'}
    return f"\033[{codes.get(code, '0')}m{text}\033[0m"

def print_step(msg): print(f'\n{color("="*54, "cyan")}\n  {msg}\n{color("="*54, "cyan")}')
def print_ok(msg):   print(f'  {color("v", "green")} {msg}')
def print_info(msg): print(f'  . {msg}')
def print_warn(msg): print(f'  {color("!", "yellow")} {msg}')

# ──────────────────────────────────────────────
# Подготовка датасета
# ──────────────────────────────────────────────

def prepare_dataset(df):
    print_step('Подготовка датасета')

    df = df[df['sentiment'].isin(VALID_LABELS)].copy()
    df = df[df['review_text'].str.len() > 10]
    df = df.drop_duplicates(subset='review_text')

    print_info(f'До балансировки: {len(df)}')
    counts = df['sentiment'].value_counts()
    print_info(str(counts.to_dict()))

    min_size = counts.min()
    if min_size < 100:
        print_warn(f'Мало данных для класса "{counts.idxmin()}": {min_size} записей.')
        print_warn('Точность модели может быть низкой.')

    parts = [
        df[df['sentiment'] == label].sample(n=min_size, random_state=42)
        for label in VALID_LABELS
    ]
    df_balanced = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    print_ok(f'После балансировки: {len(df_balanced)} ({min_size} на класс)')
    return df_balanced


# ──────────────────────────────────────────────
# Эмбеддинги
# ──────────────────────────────────────────────

def build_embeddings(df):
    print_step('Построение sentence-transformer эмбеддингов')
    print_info(f'Модель: {MODEL_NAME}')
    print_info(f'Текстов: {len(df)}  (займет несколько минут)')

    st_model = SentenceTransformer(MODEL_NAME)
    t0 = time.time()
    embeddings = st_model.encode(
        df['review_text'].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print_ok(f'Готово за {time.time()-t0:.0f} с. Размер: {embeddings.shape}')
    return embeddings


# ──────────────────────────────────────────────
# Обучение
# ──────────────────────────────────────────────

def train_models_lgbm(df, embeddings):
    print_step('Обучение LightGBM моделей')

    le = LabelEncoder()
    le.fit(VALID_LABELS)
    y_multi = le.transform(df['sentiment'])
    print_info(f'Классы: {le.classes_}')

    # Многоклассовая
    X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(
        embeddings, y_multi, test_size=0.2, random_state=42, stratify=y_multi
    )
    print_info('Обучаем многоклассовую модель...')
    t0 = time.time()
    multiclass_model = LGBMClassifier(**LGBM_PARAMS)
    multiclass_model.fit(X_tr_m, y_tr_m)
    print_ok(f'Многоклассовая готова за {time.time()-t0:.0f} с')
    y_pred_m = multiclass_model.predict(X_te_m)
    print_ok(f'Точность: {accuracy_score(y_te_m, y_pred_m):.4f}')
    print(classification_report(
        le.inverse_transform(y_te_m), le.inverse_transform(y_pred_m),
        target_names=VALID_LABELS, zero_division=0
    ))

    # Бинарная
    mask    = df['sentiment'] != 'Смешанный'
    emb_bin = embeddings[mask.values]
    y_bin   = (df[mask]['sentiment'] == 'Позитив').astype(int).values

    X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(
        emb_bin, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )
    print_info('Обучаем бинарную модель...')
    t0 = time.time()
    binary_model = LGBMClassifier(**LGBM_PARAMS)
    binary_model.fit(X_tr_b, y_tr_b)
    print_ok(f'Бинарная готова за {time.time()-t0:.0f} с')
    y_pred_b = binary_model.predict(X_te_b)
    print_ok(f'Точность: {accuracy_score(y_te_b, y_pred_b):.4f}')
    print(classification_report(
        y_te_b, y_pred_b,
        target_names=['Негатив', 'Позитив'], zero_division=0
    ))

    return binary_model, multiclass_model, le


# ──────────────────────────────────────────────
# Оценка каскада
# ──────────────────────────────────────────────

def evaluate_cascade(df, embeddings, binary_model, multiclass_model, le):
    print_step('Оценка каскадной системы')

    y_true = le.transform(df['sentiment'])
    _, emb_test, _, y_test = train_test_split(
        embeddings, y_true, test_size=0.2, random_state=42, stratify=y_true
    )

    preds    = []
    bin_used = 0

    for emb in emb_test:
        s = [emb]
        try:
            pb    = binary_model.predict_proba(s)[0]
            max_p = float(np.max(pb))
            dif_p = float(abs(pb[0] - pb[1]))
            if max_p >= THRESHOLD and dif_p >= MIN_DIFF:
                lbl = 'Негатив' if int(binary_model.predict(s)[0]) == 0 else 'Позитив'
                preds.append(le.transform([lbl])[0])
                bin_used += 1
                continue
        except Exception:
            pass
        preds.append(int(multiclass_model.predict(s)[0]))

    preds = np.array(preds)
    acc   = accuracy_score(y_test, preds)

    print_ok(f'Точность каскада: {acc:.4f}')
    print_info(f'Бинарная модель: {bin_used}/{len(emb_test)} ({bin_used/len(emb_test):.1%})')
    print()
    print(classification_report(
        le.inverse_transform(y_test), le.inverse_transform(preds),
        target_names=VALID_LABELS, zero_division=0
    ))


# ──────────────────────────────────────────────
# Сохранение
# ──────────────────────────────────────────────

def save_models(binary_model, multiclass_model):
    print_step('Сохранение моделей')
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(binary_model,     BINARY_MODEL_PATH)
    joblib.dump(multiclass_model, MULTICLASS_MODEL_PATH)
    print_ok(f'Бинарная:       {BINARY_MODEL_PATH}')
    print_ok(f'Многоклассовая: {MULTICLASS_MODEL_PATH}')
    print()
    print('  Запустите анализ:')
    print('  python analyze_sentiments.py --input content/resident_complex_reviews_report.json')


# ──────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Обучение моделей тональности на реальных отзывах из JSON',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--input', '-i', nargs='+', default=None,
        help=f'JSON-файлы для обучения.\nПо умолчанию: все .json из {CONTENT_DIR}'
    )
    parser.add_argument(
        '--dataset-only', action='store_true',
        help='Только сформировать датасет, без обучения моделей'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print()
    print(color('╔══════════════════════════════════════════════════════╗', 'cyan'))
    print(color('║     Обучение моделей на реальных отзывах (JSON)      ║', 'bold'))
    print(color('╚══════════════════════════════════════════════════════╝', 'cyan'))

    # ── Шаг 1: Парсинг ──
    if args.input:
        filepaths = [Path(p) for p in args.input]
    else:
        if not CONTENT_DIR.exists():
            sys.exit(color(
                f'Папка {CONTENT_DIR} не найдена. '
                f'Укажите файлы вручную: --input файл.json', 'red'
            ))
        filepaths = sorted(CONTENT_DIR.glob('*.json'))
        if not filepaths:
            sys.exit(color(f'В папке {CONTENT_DIR} нет .json файлов.', 'red'))

    print_step('Парсинг JSON-файлов')
    print_info(f'Файлов: {len(filepaths)}')
    for p in filepaths:
        print_info(f'  {p}')

    df_raw = load_all_json(filepaths, with_sentiment=True)
    if len(df_raw) == 0:
        sys.exit(color('Не удалось извлечь ни одного отзыва.', 'red'))
    print_ok(f'Всего отзывов: {len(df_raw)}')

    # ── Шаг 2: Подготовка ──
    df = prepare_dataset(df_raw)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False, encoding='utf-8-sig')
    print_ok(f'Балансированный датасет: {DATASET_PATH} ({len(df)} строк)')

    if args.dataset_only:
        print_ok('Режим --dataset-only: обучение пропущено.')
        return

    # ── Шаг 3–6: Эмбеддинги, обучение, оценка, сохранение ──
    embeddings = build_embeddings(df)
    binary_model, multiclass_model, le = train_models_lgbm(df, embeddings)
    evaluate_cascade(df, embeddings, binary_model, multiclass_model, le)
    save_models(binary_model, multiclass_model)


if __name__ == '__main__':
    main()