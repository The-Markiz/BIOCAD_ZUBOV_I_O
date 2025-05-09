import argparse
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, ConfusionMatrixDisplay,
    RocCurveDisplay
)
import joblib
import matplotlib.pyplot as plt
import lightgbm as lgb

# 1. Загрузка и базовая очистка данных

def load_and_clean(path: str) -> pd.DataFrame:
    """
    Загружает CSV-файл с данными SKEMPI и приводит некоторые столбцы к числовому типу.

    Args:
        path (str): Путь до CSV-файла с разделителем ";".

    Returns:
        pd.DataFrame: Очищенный датафрейм.
    """
    # Чтение CSV и вывод числа строк
    df = pd.read_csv(path, sep=';')
    print(f"[0] Всего строк в файле: {df.shape[0]}")

    # Колонки, которые должны быть числовыми, но могут содержать нечисловые значения
    columns_to_convert = [
        'Affinity_mut_parsed', 'Affinity_wt_parsed', 'Temperature',
        'dS_mut (cal mol^(-1) K^(-1))', 'dS_wt (cal mol^(-1) K^(-1))'
    ]
    # Преобразуем указанные колонки в числовой формат, нечисловые значения станут NaN
    for col in columns_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# 2. Расчет ddG (разницы свободной энергии связывания)

def calculate_ddG(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет dS в килокалориях и дельту свободной энергии ddG.

    Args:
        df (pd.DataFrame): Датафрейм с колонками Affinity и dS/Temperature.

    Returns:
        pd.DataFrame: Датафрейм с добавленными столбцами dS_mut_kcal, dS_wt_kcal и ddG.
    """
    # Перевод энтропии из кал/моль*K в ккал/моль
    df['dS_mut_kcal'] = df['dS_mut (cal mol^(-1) K^(-1))'] / 1000
    df['dS_wt_kcal'] = df['dS_wt (cal mol^(-1) K^(-1))'] / 1000

    # Формула: ddG = dG_mut - dG_wt = Affinity_mut - Affinity_wt - T * (dS_mut - dS_wt)
    df['ddG'] = (
        df['Affinity_mut_parsed'] - df['Affinity_wt_parsed']
    ) - df['Temperature'] * (
        df['dS_mut_kcal'] - df['dS_wt_kcal']
    )
    return df


# 3. Создание целевой переменной ddG_sign

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет бинарную целевую переменную ddG_sign:
    1, если ddG > 0 (снижение аффинности), иначе 0.

    Args:
        df (pd.DataFrame): Датафрейм с колонкой ddG.

    Returns:
        pd.DataFrame: Датафрейм с колонкой ddG_sign.
    """
    df['ddG_sign'] = (df['ddG'] > 0).astype(int)
    return df


# 4. Генерация дополнительных признаков

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет логарифмы аффинностей и энтальпийную разность.

    Args:
        df (pd.DataFrame): Датафрейм с исходными признаками.

    Returns:
        pd.DataFrame: Датафрейм с новыми признаками log_aff_mut, log_aff_wt, delta_H, delta_S.
    """
    # Логарифм аффинности; игнорируем нули
    df['log_aff_mut'] = np.log(df['Affinity_mut_parsed'].replace(0, np.nan))
    df['log_aff_wt'] = np.log(df['Affinity_wt_parsed'].replace(0, np.nan))

    # Разница энтальпий (dH_mut - dH_wt)
    df['delta_H'] = (
        pd.to_numeric(df['dH_mut (kcal mol^(-1))'], errors='coerce') -
        pd.to_numeric(df['dH_wt (kcal mol^(-1))'], errors='coerce')
    )
    # Разница энтропий (уже в ккал)
    df['delta_S'] = df['dS_mut_kcal'] - df['dS_wt_kcal']
    return df


# 5. Удаление выбросов по IQR и Z-score

def filter_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Фильтрует выбросы в указанных колонках:
    - Отсекает значения за пределами [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    - Отсекает точки с |Z| >= 3

    Args:
        df (pd.DataFrame): Датафрейм с признаками.
        cols (list): Список названий колонок для фильтрации.

    Returns:
        pd.DataFrame: Отфильтрованный датафрейм.
    """
    mask = pd.Series(True, index=df.index)
    for c in cols:
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        # Z-score для проверки выбросов
        z = np.abs(zscore(df[c].fillna(s.median())))
        # Обновляем маску, остается True только для непиковых значений
        mask &= df[c].between(lb, ub) & (z < 3)
    return df[mask]


# 6. Построение пайплайна предобработки и классификатора

def build_pipeline(numeric_feats: list, categorical_feats: list) -> Pipeline:
    """
    Создает sklearn Pipeline с:
    - заполнением пропусков и стандартизацией для числовых признаков
    - one-hot кодированием для категориальных признаков
    - классификатором LGBM

    Args:
        numeric_feats (list): Список числовых признаков.
        categorical_feats (list): Список категориальных признаков.

    Returns:
        Pipeline: Полный пайплайн.
    """
    # Пайплайн для числовых признаков
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # медианная замена пропусков
        ('scaler', StandardScaler())                    # стандартизация
    ])

    # Комбинированный препроцессор: числовые + категориальные
    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_feats),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_feats)
    ])

    # Модель LightGBM
    model = LGBMClassifier(
        random_state=42,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=-1
    )

    # Полный пайплайн: предобработка + классификатор
    return Pipeline([('prep', preprocessor), ('clf', model)])


# 7. Обучение модели, подбор гиперпараметров и визуализация результатов

def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test, param_grid=None):
    """
    Тренирует и оценивает модель:
    - При наличии param_grid выполняет GridSearchCV по ROC AUC
    - Выводит accuracy, classification_report и ROC AUC
    - Строит матрицу ошибок, ROC-кривую и график важности признаков

    Args:
        pipeline: sklearn Pipeline.
        X_train, y_train, X_test, y_test: Наборы данных.
        param_grid (dict, optional): Сетка параметров для GridSearchCV.

    Returns:
        Обученный sklearn Pipeline или лучший estimator.
    """
    import matplotlib.gridspec as gridspec
    from operator import itemgetter

    # Поиск лучших параметров, если задано
    if param_grid:
        search = GridSearchCV(
            pipeline, param_grid, cv=5,
            scoring='roc_auc', n_jobs=-1, verbose=1
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print("Лучшие параметры:", search.best_params_)
        print(f"CV ROC AUC: {search.best_score_:.3f}")
    else:
        model = pipeline.fit(X_train, y_train)

    # Предсказания и вероятности
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Метрики
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {auc:.4f}")
    print(f"Кросс-валидация (accuracy): {scores.mean():.3f} ± {scores.std():.3f}")

    # Извлекаем названия признаков после препроцессинга
    feature_names = model.named_steps['prep'].get_feature_names_out()
    importances = model.named_steps['clf'].feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]
    top_names = itemgetter(*top_indices)(feature_names)
    top_importances = importances[top_indices]

    # Визуализация результатов
    fig = plt.figure(figsize=(14, 10))
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

    # 1) Матрица ошибок
    ax0 = fig.add_subplot(spec[0, 0])
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax0)
    ax0.set_title("Матрица ошибок")

    # 2) ROC-кривая
    ax1 = fig.add_subplot(spec[0, 1])
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax1)
    ax1.set_title("ROC-кривая")

    # 3) Важность признаков
    ax2 = fig.add_subplot(spec[1, :])
    ax2.barh(range(len(top_importances)), top_importances[::-1], align='center')
    ax2.set_yticks(range(len(top_names)))
    ax2.set_yticklabels(top_names[::-1])
    ax2.set_title("Важность признаков")
    ax2.invert_yaxis()

    fig.suptitle("Оценка модели", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return model


# 8. Точка входа скрипта

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train ddG_sign classifier on SKEMPI data."
    )
    parser.add_argument(
        "--data", type=str, default="skempi_v2.csv",
        help="Path to SKEMPI CSV"
    )
    parser.add_argument(
        "--model", type=str, default="ddg_pipeline.pkl",
        help="Output model file"
    )
    parser.add_argument(
        "--no-grid", action='store_true',
        help="Disable hyperparam search"
    )
    args = parser.parse_args()

    # Загружаем и готовим данные
    df = load_and_clean(args.data)
    df = df.copy()
    print(f"[1] После загрузки: {df.shape[0]} строк")

    # Вычисляем ddG и целевую переменную
    df = calculate_ddG(df)
    df = create_target(df)
    df = feature_engineering(df)

    # Отбираем признаки и фильтруем выбросы
    feats = ['log_aff_mut', 'log_aff_wt', 'delta_H', 'delta_S', 'ddG']
    df = filter_outliers(df, feats)
    print(f"[2] После фильтрации выбросов: {df.shape[0]} строк")

    # Разделяем на числовые и категориальные признаки
    numeric_feats = feats
    categorical_feats = ['Method'] if 'Method' in df.columns else []
    X = df[numeric_feats + categorical_feats]
    y = df['ddG_sign']

    # Разбиение на обучающую и тестовую выборку
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[3] Train: {X_train.shape[0]} / Test: {X_test.shape[0]}")

    # Создаем пайплайн и задаем сетку параметров
    pipeline = build_pipeline(numeric_feats, categorical_feats)
    param_grid = None if args.no_grid else {
        'clf__num_leaves': [31, 50],
        'clf__learning_rate': [0.01, 0.05],
        'clf__min_child_samples': [5, 10]
    }

    # Обучаем модель, оцениваем и сохраняем
    model = train_and_evaluate(
        pipeline, X_train, y_train, X_test, y_test, param_grid
    )
    joblib.dump(model, args.model)
    print(f"Модель сохранена в {args.model}")
