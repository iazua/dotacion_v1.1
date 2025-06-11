import pandas as pd
import numpy as np
import holidays


def assign_turno(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna turnos basados en la hora del día.

    Args:
        df: DataFrame que debe contener columna 'HORA' (valores 9-21)

    Returns:
        DataFrame con columna 'turno' añadida:
          0: Fuera de rango o NaN
          1: 9-11
          2: 12-14
          3: 15-17
          4: 18-21
    """
    # Definimos los intervalos horarios
    bins = [8, 11, 14, 17, 21]
    labels = [1, 2, 3, 4]

    # Asignamos turnos usando pd.cut
    df['turno'] = pd.cut(
        df['HORA'],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True
    )

    # Convertimos a int y manejamos valores fuera de rango
    df['turno'] = df['turno'].cat.add_categories([0]).fillna(0).astype(int)
    return df


def prepare_features(
        df: pd.DataFrame,
        target: str,
        is_prediction: bool = False,
        include_time_features: bool = False
):
    """
    Prepara features para modelamiento, manteniendo consistencia entre entrenamiento y predicción.

    Args:
        df: DataFrame de entrada
        target: Nombre de la columna objetivo
        is_prediction: Si es True, no requiere la columna target
        include_time_features: Si es True, incluye 'turno' y 'HORA' como features

    Returns:
        X: DataFrame con features
        y: Serie con target (np.nan si is_prediction=True)
    """
    df = df.copy()

    # Validación y conversión de tipos
    df['FECHA'] = pd.to_datetime(df['FECHA'])

    # Procesamiento de hora/turno si existe la columna HORA
    if 'HORA' in df.columns:
        df['HORA'] = df['HORA'].fillna(-1)  # Valor fuera de rango para horas inválidas
        df = assign_turno(df)

    # Ordenamos por fecha (y hora si existe)
    sort_cols = ['FECHA', 'HORA'] if 'HORA' in df.columns else ['FECHA']
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # --------------------------------------------------
    # Features temporales básicas (siempre incluídas)
    # --------------------------------------------------
    df["year"] = df["FECHA"].dt.year
    df["month"] = df["FECHA"].dt.month
    df["day"] = df["FECHA"].dt.day
    df["weekday"] = df["FECHA"].dt.weekday
    df["dayofyear"] = df["FECHA"].dt.dayofyear
    df["weekofyear"] = df["FECHA"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Feriados chilenos
    min_year = df['FECHA'].min().year - 1
    max_year = df['FECHA'].max().year + 2
    chile_holidays = holidays.Chile(years=range(min_year, max_year))
    df['is_holiday'] = df['FECHA'].dt.date.isin(chile_holidays).astype(int)

    # --------------------------------------------------
    # Features de rezagos y medias móviles
    # --------------------------------------------------
    lag_cols = ["T_AO", "T_AO_VENTA", "T_VISITAS", "DOTACION"]
    for col in lag_cols:
        # Asegurar que la columna existe
        if col not in df.columns:
            df[col] = np.nan

        # Crear features derivadas
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_rolling7"] = df[col].shift(1).rolling(7, min_periods=1).mean()
        df[f"{col}_rolling14"] = df[col].shift(1).rolling(14, min_periods=1).mean()

    # --------------------------------------------------
    # Definición final de features a incluir
    # --------------------------------------------------
    features = [
        "year", "month", "day", "weekday", "dayofyear", "weekofyear",
        "is_weekend", "is_holiday",
        "T_AO_lag1", "T_AO_rolling7", "T_AO_rolling14",
        "T_AO_VENTA_lag1", "T_AO_VENTA_rolling7", "T_AO_VENTA_rolling14",
        "T_VISITAS_lag1", "T_VISITAS_rolling7", "T_VISITAS_rolling14",
        "DOTACION_lag1", "DOTACION_rolling7", "DOTACION_rolling14"
    ]

    # Añadir features horarias si se solicita y existen
    if include_time_features:
        if 'turno' in df.columns:
            features.append('turno')
        if 'HORA' in df.columns:
            features.append('HORA')

    # --------------------------------------------------
    # Preparación de X e y
    # --------------------------------------------------
    # Asegurar que todas las features existan en el DataFrame
    for feat in features:
        if feat not in df.columns:
            df[feat] = 0  # Rellenar con 0 en lugar de NaN para predict

    if is_prediction:
        X = df[features].copy()
        # Imputación simple para predicción
        X = X.fillna(0)
        y = pd.Series([np.nan] * len(df), index=df.index)
    else:
        if target not in df.columns:
            raise ValueError(f"Columna objetivo '{target}' no encontrada")

        # Eliminar filas con target faltante
        valid_rows = df[target].notna()
        X = df.loc[valid_rows, features].copy()
        y = df.loc[valid_rows, target]

        # Imputación para entrenamiento (podrías usar otra estrategia aquí)
        X = X.fillna(0)

    return X, y