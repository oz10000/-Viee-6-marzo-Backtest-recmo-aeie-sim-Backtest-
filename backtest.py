#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BACKTEST MULTI-TP/SL PARA ESTRATEGIA 3M (BTC, SOL)
- Barrido de TP y SL en un rango definido
- Incluye comisiones y análisis de apalancamiento máximo seguro
- Genera ranking de combinaciones óptimas
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

SYMBOLS = ["BTCUSDT", "SOLUSDT"]
TIMEFRAME = "3m"
LOOKBACK = 1000          # velas históricas para indicadores
LOOKAHEAD = 50           # velas máximas para esperar TP/SL
EMA_SPAN = 20
DEVIATION_THRESHOLD = 0.003   # 0.3% de desviación para entrada

# Rango de TP y SL (en fracción, ej. 0.001 = 0.1%)
TP_VALUES = np.arange(0.001, 0.011, 0.001)    # 0.1% a 1.0%
SL_VALUES = np.arange(0.001, 0.011, 0.001)    # 0.1% a 1.0%

COMMISSION = 0.0002      # 0.02% por operación (taker)

# Fechas para backtest (3 meses)
BACKTEST_START = "2025-01-01"
BACKTEST_END = "2025-04-01"

# Límite de apalancamiento máximo del exchange (por ejemplo, 100x)
MAX_LEVERAGE_EXCHANGE = 100

# ==========================================================
# FUNCIONES DE DESCARGA DE DATOS (BINANCE)
# ==========================================================

def fetch_binance_klines(symbol, interval, start_str, end_str):
    """
    Descarga velas de Binance Spot (API pública) en el rango de fechas.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_str, "%Y-%m-%d").timestamp() * 1000)
    
    all_klines = []
    current_start = start_ts
    limit = 1000  # máximo por request
    
    while current_start < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ts,
            "limit": limit
        }
        try:
            resp = requests.get(base_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            all_klines.extend(data)
            # Actualizar startTime al timestamp de la última vela + 1 ms
            current_start = data[-1][0] + 1
            time.sleep(0.1)  # evitar rate limit
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            break
    
    if not all_klines:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    return df

# ==========================================================
# INDICADORES
# ==========================================================

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

# ==========================================================
# CLASE PARA SIMULAR OPERACIONES CON MÚLTIPLES TP/SL
# ==========================================================

class MultiTPSLBacktest:
    def __init__(self, df, tp_values, sl_values, commission, lookahead=50):
        """
        df: DataFrame con columnas 'timestamp','open','high','low','close'
        tp_values: lista de take profits (fracción)
        sl_values: lista de stop losses (fracción)
        commission: comisión fraccional por operación
        lookahead: número máximo de velas a futuro para evaluar salida
        """
        self.df = df
        self.tp_values = tp_values
        self.sl_values = sl_values
        self.commission = commission
        self.lookahead = lookahead
        
        # Precalcular EMA y desviación
        self.df['ema'] = ema(self.df['close'], EMA_SPAN)
        self.df['deviation'] = (self.df['close'] - self.df['ema']) / self.df['ema']
        
        # Identificar señales de entrada
        self.entries = self._find_entries()
        print(f"Señales de entrada encontradas: {len(self.entries)}")
        
    def _find_entries(self):
        """Encuentra índices donde |deviation| > umbral (estrategia mean reversion)"""
        entries = []
        for i in range(30, len(self.df) - self.lookahead):  # dejar margen para indicadores y lookahead
            if abs(self.df.loc[i, 'deviation']) > DEVIATION_THRESHOLD:
                direction = 'SHORT' if self.df.loc[i, 'deviation'] > 0 else 'LONG'
                entries.append({
                    'idx': i,
                    'timestamp': self.df.loc[i, 'timestamp'],
                    'price': self.df.loc[i, 'close'],
                    'direction': direction
                })
        return entries
    
    def _simulate_trade_outcome(self, entry_idx, entry_price, direction, tp, sl):
        """
        Simula el resultado de una operación dados tp y sl (fracciones).
        Retorna: resultado bruto (fracción de entrada, positivo para ganancia),
                 tiempo de salida en velas, y si se alcanzó tp, sl o final.
        """
        start_idx = entry_idx + 1
        end_idx = min(entry_idx + self.lookahead, len(self.df) - 1)
        
        # Determinar precios objetivo
        if direction == 'LONG':
            tp_price = entry_price * (1 + tp)
            sl_price = entry_price * (1 - sl)
        else:  # SHORT
            tp_price = entry_price * (1 - tp)
            sl_price = entry_price * (1 + sl)
        
        # Recorrer velas futuras
        for j in range(start_idx, end_idx + 1):
            high = self.df.loc[j, 'high']
            low = self.df.loc[j, 'low']
            
            if direction == 'LONG':
                if high >= tp_price:
                    return tp, j - entry_idx, 'tp'
                if low <= sl_price:
                    return -sl, j - entry_idx, 'sl'
            else:
                if low <= tp_price:
                    return tp, j - entry_idx, 'tp'
                if high >= sl_price:
                    return -sl, j - entry_idx, 'sl'
        
        # Si no se alcanzó ni tp ni sl, cerramos al final del lookahead
        final_price = self.df.loc[end_idx, 'close']
        if direction == 'LONG':
            result = (final_price - entry_price) / entry_price
        else:
            result = (entry_price - final_price) / entry_price
        return result, end_idx - entry_idx, 'final'
    
    def run_all_combinations(self):
        """
        Ejecuta la simulación para todas las combinaciones de tp y sl.
        Retorna un dict con resultados por combinación.
        """
        results = {}
        # Pre-crear estructura para resultados
        for tp in self.tp_values:
            for sl in self.sl_values:
                results[(tp, sl)] = {
                    'trades': [],
                    'results_net': [],
                    'exit_types': [],
                    'durations': []
                }
        
        # Para cada entrada, calcular resultados para todas las combinaciones
        total_entries = len(self.entries)
        for idx, entry in enumerate(self.entries):
            if (idx + 1) % 10 == 0:
                print(f"Procesando entrada {idx+1}/{total_entries}")
            
            entry_idx = entry['idx']
            entry_price = entry['price']
            direction = entry['direction']
            
            for tp in self.tp_values:
                for sl in self.sl_values:
                    result_bruto, duration, exit_type = self._simulate_trade_outcome(
                        entry_idx, entry_price, direction, tp, sl
                    )
                    # Calcular resultado neto de comisiones
                    # Se paga comisión al abrir y al cerrar
                    result_net = result_bruto - 2 * self.commission
                    results[(tp, sl)]['trades'].append({
                        'entry_idx': entry_idx,
                        'direction': direction,
                        'result_bruto': result_bruto,
                        'result_net': result_net,
                        'duration': duration,
                        'exit_type': exit_type
                    })
                    results[(tp, sl)]['results_net'].append(result_net)
                    results[(tp, sl)]['exit_types'].append(exit_type)
                    results[(tp, sl)]['durations'].append(duration)
        
        return results
    
    def compute_metrics(self, results_dict):
        """
        A partir del dict de resultados, calcula métricas por combinación.
        """
        metrics = []
        for (tp, sl), data in results_dict.items():
            res_net = np.array(data['results_net'])
            n_trades = len(res_net)
            if n_trades == 0:
                continue
            
            wins = res_net[res_net > 0]
            losses = res_net[res_net <= 0]
            win_rate = len(wins) / n_trades if n_trades > 0 else 0
            avg_win = np.mean(wins) if len(wins) > 0 else 0
            avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0
            profit_factor = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 else np.inf
            expectancy_net = np.mean(res_net)  # en fracción del precio
            # Calcular expectancy en R (múltiplos del riesgo = sl)
            # Nota: el riesgo es sl, pero la pérdida neta puede ser diferente por comisión
            # Definimos R como sl (sin comisión) para mantener consistencia
            r_values = res_net / sl  # esto da el múltiplo de R (aproximado)
            expectancy_R = np.mean(r_values)
            
            # Sharpe ratio de los retornos netos (asumiendo rf=0)
            if n_trades > 1:
                sharpe = np.mean(res_net) / np.std(res_net) * np.sqrt(252*480)  # ajuste a anual? mejor no
                # Simplificado: Sharpe de los retornos por trade (no time-weighted)
                sharpe = np.mean(res_net) / np.std(res_net) if np.std(res_net) > 0 else 0
            else:
                sharpe = 0
            
            # Máximo drawdown de la curva de equity (simulada como suma de resultados)
            equity = np.cumsum(res_net)
            rolling_max = np.maximum.accumulate(equity)
            drawdown = (equity - rolling_max) / (rolling_max + 1e-9)
            max_dd = np.min(drawdown) if len(drawdown) > 0 else 0
            
            # Apalancamiento máximo seguro basado en SL
            # Para evitar liquidación, necesitamos SL < 1/L => L < 1/SL
            max_leverage_safe = min(int(1 / sl), MAX_LEVERAGE_EXCHANGE) if sl > 0 else 0
            
            metrics.append({
                'tp': tp,
                'sl': sl,
                'n_trades': n_trades,
                'win_rate': win_rate,
                'avg_win_net': avg_win,
                'avg_loss_net': avg_loss,
                'profit_factor': profit_factor,
                'expectancy_net': expectancy_net,
                'expectancy_R': expectancy_R,
                'sharpe': sharpe,
                'max_drawdown': max_dd,
                'max_leverage_safe': max_leverage_safe,
                # Distribución de salidas
                'pct_tp': data['exit_types'].count('tp') / n_trades,
                'pct_sl': data['exit_types'].count('sl') / n_trades,
                'pct_final': data['exit_types'].count('final') / n_trades,
                'avg_duration': np.mean(data['durations'])
            })
        
        return pd.DataFrame(metrics)

# ==========================================================
# EJECUCIÓN PRINCIPAL
# ==========================================================

def main():
    print("="*70)
    print("BACKTEST MULTI-TP/SL PARA ESTRATEGIA 3M")
    print("="*70)
    
    all_results = []
    
    for symbol in SYMBOLS:
        print(f"\n📥 Descargando datos para {symbol}...")
        df = fetch_binance_klines(symbol, TIMEFRAME, BACKTEST_START, BACKTEST_END)
        if df.empty:
            print(f"❌ No se pudieron obtener datos para {symbol}")
            continue
        print(f"✅ {len(df)} velas descargadas")
        
        bt = MultiTPSLBacktest(df, TP_VALUES, SL_VALUES, COMMISSION, LOOKAHEAD)
        results_dict = bt.run_all_combinations()
        metrics_df = bt.compute_metrics(results_dict)
        metrics_df['symbol'] = symbol
        all_results.append(metrics_df)
    
    if not all_results:
        print("No hay resultados.")
        return
    
    # Combinar resultados de todos los símbolos
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Ordenar por expectancy_R (mejor)
    final_df = final_df.sort_values('expectancy_R', ascending=False)
    
    # Guardar a CSV
    final_df.to_csv('backtest_tpsl_results.csv', index=False)
    print("\n" + "="*70)
    print("RESULTADOS GLOBALES (TOP 20 POR EXPECTANCY_R)")
    print("="*70)
    print(final_df.head(20).to_string(index=False))
    
    # También podemos generar heatmaps o gráficos, pero aquí solo mostramos texto.
    
    # Análisis adicional: mejor combinación por símbolo
    print("\n" + "="*70)
    print("MEJOR COMBINACIÓN POR SÍMBOLO (según expectancy_R)")
    print("="*70)
    for symbol in SYMBOLS:
        sub = final_df[final_df['symbol'] == symbol]
        if not sub.empty:
            best = sub.iloc[0]
            print(f"{symbol}: TP={best['tp']:.3%}, SL={best['sl']:.3%}, "
                  f"WinRate={best['win_rate']:.2%}, Expectancy_R={best['expectancy_R']:.4f}, "
                  f"ProfitFactor={best['profit_factor']:.2f}")
    
    print("\n✅ Backtest completado. Resultados guardados en backtest_tpsl_results.csv")

if __name__ == "__main__":
    main()
