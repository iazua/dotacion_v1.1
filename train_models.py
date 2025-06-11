import os
import pandas as pd
import joblib
import numpy as np
from datetime import timedelta
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error
from preprocessing import prepare_features, assign_turno
from utils import estimar_dotacion_optima, estimar_parametros_efectividad, calcular_efectividad

# Configuración general: usar carpeta models2 para consistencia con la app
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

HORIZON_DAYS = 7
HOURS_RANGE = list(range(9, 22))  # 9–21 horario completo


def load_data(file_path: str) -> pd.DataFrame:
    """Carga datos con columna HORA obligatoria y parsea FECHA"""
    df = pd.read_excel(file_path)
    if 'HORA' not in df.columns:
        raise ValueError("La columna 'HORA' es obligatoria para modelado horario.")
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    return df


def train_model_per_branch(
    df: pd.DataFrame,
    target: str,
    param_grid: dict = None
):
    """
    Entrena un XGBRegressor por sucursal para el target dado.
    Usa TimeSeriesSplit y GridSearchCV.
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    branches = df['COD_SUC'].unique()
    for branch in branches:
        df_branch = df[df['COD_SUC'] == branch].copy()
        X, y = prepare_features(df_branch, target, is_prediction=False)
        if len(X) < max(param_grid['n_estimators']) * 2:
            print(f"⚠️ Datos insuficientes para sucursal {branch}-{target}, se omite.")
            continue
        tscv = TimeSeriesSplit(n_splits=5)
        gs = GridSearchCV(
            XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
            param_grid,
            scoring='neg_mean_absolute_error',
            cv=tscv,
            n_jobs=-1,
            verbose=1
        )
        gs.fit(X, y)
        best = gs.best_estimator_
        path = os.path.join(MODEL_DIR, f"predictor_{target}_{branch}.pkl")
        joblib.dump(best, path)
        print(f"✅ Guardado modelo {target} sucursal {branch} con params {gs.best_params_}")


def predict_future_per_branch(
    df: pd.DataFrame,
    branch: int,
    targets: list,
    P_obj: float = 0.8
) -> pd.DataFrame:
    """
    Genera predicciones por hora para los próximos HORIZON_DAYS días,
    calculando T_AO_VENTA_req, P_EFECTIVIDAD_req y DOTACION_opt.
    """
    df_hist = df[df['COD_SUC'] == branch].copy()
    df_hist['FECHA'] = pd.to_datetime(df_hist['FECHA'])
    last_date = df_hist['FECHA'].max()
    rows = [
        {'FECHA': last_date + timedelta(days=d), 'HORA': hr, 'COD_SUC': branch}
        for d in range(1, HORIZON_DAYS + 1)
        for hr in HOURS_RANGE
    ]
    df_fut = pd.DataFrame(rows)
    df_fut = assign_turno(df_fut)

    df_all = pd.concat([df_hist, df_fut], ignore_index=True, sort=False)
    mask = df_all.index >= len(df_hist)
    df_out = df_fut.copy().reset_index(drop=True)
    df_out['COD_SUC'] = branch

    # Predecir cada target con su propio bloque de features
    for t in targets:
        model_path = os.path.join(MODEL_DIR, f"predictor_{t}_{branch}.pkl")
        if not os.path.exists(model_path):
            df_out[f"{t}_pred"] = np.nan
            continue
        # preparar features específicas
        X_all_t, _ = prepare_features(df_all, target=t, is_prediction=True)
        X_all_t = X_all_t.drop(columns=["turno","HORA"], errors='ignore')
        X_fut_t = X_all_t[mask].reset_index(drop=True)
        model = joblib.load(model_path)
        df_out[f"{t}_pred"] = model.predict(X_fut_t)

    # Ventas necesarias y efectividad
    df_out['T_AO_VENTA_req'] = df_out['T_AO_pred'] * P_obj
    df_out['P_EFECTIVIDAD_req'] = calcular_efectividad(
        df_out['T_AO_pred'], df_out['T_AO_VENTA_req']
    )

    # Parámetros históricos de efectividad
    hist_params_df = df_hist[['DOTACION','T_AO','T_AO_VENTA']].dropna()
    params = None
    if len(hist_params_df) >= 3:
        params = estimar_parametros_efectividad(hist_params_df)

    dots = []
    for _, row in df_out.iterrows():
        dot_opt, _ = estimar_dotacion_optima(
            t_ao_preds=[row['T_AO_pred']],
            t_ao_venta_preds=[row['T_AO_VENTA_req']],
            efectividad_deseada=P_obj,
            params_efectividad=params
        )
        dots.append(dot_opt)
    df_out['DOTACION_opt'] = dots

    return df_out


if __name__ == '__main__':
    df = load_data('data/DOTACION_EFECTIVIDAD.xlsx')
    print('✅ Datos cargados')

    for target in ['T_VISITAS', 'T_AO']:
        print(f"🔄 Entrenando modelos para {target}...")
        train_model_per_branch(df, target)

    branches = df['COD_SUC'].unique()
    for suc in branches:
        print(f"\n🔮 Predicción para sucursal {suc}")
        df_pred = predict_future_per_branch(df, suc, ['T_VISITAS','T_AO'], P_obj=0.8)
        print(df_pred.head(20).to_string(index=False))
