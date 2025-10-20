#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script per generare uno scatter plot Pareto da file CSV multipli.
Confronta la latenza vs. il rate di diversi microcontrollori e settings.
- Punti con "modelinram" = simbolo pieno
- Punti senza "modelinram" = simbolo cavo (solo contorno colorato)
- Colore del punto = settings
- Simbolo = tipo di micro
- Fronte Pareto = linea nera che connette i punti ottimali
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

# Dimensione dei punti scatter
scatter_point_size = 100

# -----------------------------
# Funzione per calcolare il fronte di Pareto
# -----------------------------
def pareto_front(df, x_col, y_col):
    """
    Calcola il fronte di Pareto da massimizzare x e minimizzare y.
    Parametri:
    - df: DataFrame contenente tutti i punti
    - x_col: nome colonna asse X (rate)
    - y_col: nome colonna asse Y (latenza)
    Ritorna:
    - DataFrame dei punti del fronte Pareto (x_col, y_col)
    Procedura:
    1. Rimuove valori NaN
    2. Per ogni valore x mantiene solo y minima
    3. Ordina x decrescente e mantiene solo punti con y decrescente (miglioramento)
    """
    # Copia dei dati e rimozione NaN
    d = df[[x_col, y_col]].dropna().copy()

    # Step 1: per ogni x, tieni solo y minima
    d = d.groupby(x_col, as_index=False)[y_col].min()

    # Step 2: ordina per x decrescente (rate alto a sinistra)
    d = d.sort_values(by=[x_col], ascending=False)

    # Step 3: scansione per filtrare y decrescenti
    front_x, front_y = [], []
    best_y = np.inf
    for _, row in d.iterrows():
        x, y = row[x_col], row[y_col]
        if y < best_y:
            front_x.append(x)
            front_y.append(y)
            best_y = y

    return pd.DataFrame({x_col: front_x, y_col: front_y})


# -----------------------------
# Funzione principale di plot
# -----------------------------
def plot_pareto_scatter(files, output_png):
    """
    Genera uno scatter plot Pareto a partire da una lista di CSV.
    Parametri:
    - files: lista di percorsi CSV
    - output_png: percorso file PNG di output
    """
    # Colonne standard
    x_col = "Rate_DL_py_load_test_tflite" # TODO: rimetti MCU
    y_col = "mean_tot_latency_fast"
    env_col = "env_name"
    settings_col = "end_folder_Training_Size_dd_epochs"
    err_cols = ("Error_does_not_fit", "Error_model_in_ram")

    # Lista per concatenare tutti i DataFrame
    dfs = []

    # Lettura e filtraggio CSV
    for file in files:
        if not os.path.exists(file):
            print(f"[AVVISO] File non trovato: {file}")
            continue

        df = pd.read_csv(file)

        # Controllo colonne essenziali
        if not set([x_col, y_col, env_col, settings_col]).issubset(df.columns):
            print(f"[AVVISO] Colonne mancanti in {file}")
            continue

        # Filtra righe con errori
        df = df[(df[err_cols[0]] == 0) & (df[err_cols[1]] == 0)]
        df = df.dropna(subset=[x_col, y_col])
        dfs.append(df)

    if not dfs:
        print("Nessun file CSV valido trovato.")
        return

    # Concatenazione dati da tutti i CSV
    data_all = pd.concat(dfs, ignore_index=True)

    # -----------------------------
    # Normalizzazione nomi micro
    # -----------------------------
    # Per unificare eventuali env_name simili con suffisso "modelinram"
    def normalize_micro(name: str) -> str:
        return str(name).replace("-modelinram", "").replace("_modelinram", "")

    # Colonna micro_base senza suffisso
    data_all["micro_base"] = data_all[env_col].astype(str).apply(normalize_micro)

    # Colonna binaria: 1 se "modelinram" presente, 0 altrimenti
    data_all["modelinram"] = data_all[env_col].str.contains("modelinram", case=False).astype(int)

    # Micro e settings unici
    unique_micro = sorted(data_all["micro_base"].unique())
    unique_settings = sorted(data_all[settings_col].astype(str).unique())

    # -----------------------------
    # Assegnazione simboli e colori
    # -----------------------------
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>', 'h', 'H']
    micro_to_marker = {m: markers[i % len(markers)] for i, m in enumerate(unique_micro)}

    # Colori tab20 per settings
    cmap = plt.get_cmap("tab20")
    settings_to_color = {s: cmap(i % cmap.N) for i, s in enumerate(unique_settings)}

    # Creazione figura
    fig, ax = plt.subplots(figsize=(9, 7))

    # -----------------------------
    # Plot dei punti scatter
    # -----------------------------
    for s in unique_settings:
        df_s = data_all[data_all[settings_col].astype(str) == s]
        for m in unique_micro:
            df_sm = df_s[df_s["micro_base"] == m]
            if df_sm.empty:
                continue

            # Punti "modelinram" = simbolo pieno
            df_full = df_sm[df_sm["modelinram"] == 1]
            if not df_full.empty:
                ax.scatter(df_full[x_col], df_full[y_col],
                           s=scatter_point_size, marker=micro_to_marker[m],
                           facecolor=settings_to_color[s],
                           edgecolor="k", linewidths=0.5, alpha=0.80)

            # Punti senza "modelinram" = simbolo cavo
            df_hollow = df_sm[df_sm["modelinram"] == 0]
            if not df_hollow.empty:
                ax.scatter(df_hollow[x_col], df_hollow[y_col],
                           s=scatter_point_size, marker=micro_to_marker[m],
                           facecolor="none", edgecolor=settings_to_color[s],
                           linewidths=1.4, alpha=0.80)

    # -----------------------------
    # Fronte Pareto
    # -----------------------------
    pf = pareto_front(data_all, x_col, y_col)
    if not pf.empty:
        ax.plot(pf[x_col], pf[y_col], color="black", linestyle="-", linewidth=1.0, alpha=0.80, label="Fronte Pareto")

    # -----------------------------
    # Configurazione assi e titolo
    # -----------------------------
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Pareto Scatter Plot: Rate vs Latenza")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Inverti asse X (rate alti a sinistra, ottimo in basso a sinistra)
    plt.gca().invert_xaxis()

    # -----------------------------
    # Legende multiple
    # -----------------------------
    # Simboli micro
    micro_legend = [
        Line2D([0], [0], marker=micro_to_marker[m], color='w', markerfacecolor='gray',
               markeredgecolor='k', linewidth=0, markersize=8, label=str(m))
        for m in unique_micro
    ]
    micro_legend.append(Line2D([0], [0], color='k', linewidth=1.2, label="Fronte Pareto"))
    leg1 = ax.legend(handles=micro_legend, title="Micro (env_name)", loc="upper center", frameon=True)
    ax.add_artist(leg1)

    # Colori settings
    settings_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=settings_to_color[s],
               markeredgecolor='k', linewidth=0, markersize=8, label=str(s))
        for s in unique_settings
    ]
    leg2 = ax.legend(handles=settings_legend, title="Settings (end_folder_Training_Size_dd_epochs)", loc="upper right", frameon=True)
    ax.add_artist(leg2)

    # Legenda per variante micro e fronte Pareto
    fill_legend = [
        Line2D([0], [0], marker='o', color='k', markerfacecolor='k', markeredgecolor='k',
               linewidth=0, markersize=7, label="modelinram"),
        Line2D([0], [0], marker='o', color='k', markerfacecolor='none', markeredgecolor='k',
               linewidth=1.2, markersize=7, label="no modelinram")
    ]
    ax.legend(handles=fill_legend, title="Variante", loc="center right", frameon=True)

    # -----------------------------
    # Salvataggio figura
    # -----------------------------
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    #plt.show()
    print(f"Salvato grafico in {output_png}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    files = [
        # Inserire i CSV qui
        "ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_esp32-s2-saola-tflm.csv"
    ]
    plot_pareto_scatter(files, "pareto_plot.png")
