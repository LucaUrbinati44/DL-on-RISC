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
import textwrap  # Per andare a capo nelle legende lunghe
import re
import subprocess

def is_windows():
    return 1 if os.name == 'nt' else 0
ISWINDOWS = is_windows()

base_folder = '/mnt/c/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/'
if ISWINDOWS:
    #base_folder = subprocess.check_output(["wslpath", "-w", base_folder]).decode().strip()
    base_folder = subprocess.check_output(["wsl", "wslpath", "-w", base_folder]).decode().strip()
    print(base_folder)
output_folder = os.path.join(base_folder, 'Output_Python')
pareto_plot_folder = os.path.join(base_folder, 'Pareto_plot')
mcu_profiling_folder = os.path.join(output_folder, 'Profiling_Search_MCU')
output_figure = os.path.join(pareto_plot_folder, "pareto_plot_full.png")
output_figure_zoom = os.path.join(pareto_plot_folder, "pareto_plot_zoom.png")

files = [
    # Inserire i CSV qui
    os.path.join(pareto_plot_folder, "ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_esp32-s2-saola-tflm.csv"),
    #os.path.join(pareto_plot_folder, "ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_nucleo-h753zi.csv"),
    os.path.join(pareto_plot_folder, "ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_nucleo-h753zi_fake.csv"),
    os.path.join(pareto_plot_folder, "ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_fake_esp32-s2-saola-tflm.csv"),
    os.path.join(pareto_plot_folder, "ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_fake2_esp32-s2-saola-tflm.csv"),
]

files = [
    # Inserire i CSV qui
    os.path.join(mcu_profiling_folder, "profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_pico.csv"),
    os.path.join(mcu_profiling_folder, "profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_esp32-s2-saola-tflm.csv"),
    os.path.join(mcu_profiling_folder, "profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_nucleo-f446ze.csv"),
    os.path.join(mcu_profiling_folder, "profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_nucleo-h753zi.csv"),
]

# Imposta font size per ogni elemento del grafico
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes: for legend title
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Dimensione dei punti scatter
markersize = 10
scatter_point_size = markersize**2

# -----------------------------
# Funzione per calcolare il fronte di Pareto
# -----------------------------
def pareto_front(df, x_col, y_col):
    """
    Calcola il fronte di Pareto (massimizza x, minimizza y).
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
# Funzione per generare scatter plot
# -----------------------------
def plot_pareto_scatter(files, output_png, zoom=False):
    """
    Genera uno scatter plot Pareto a partire da una lista di CSV (completo o zoom sul fronte Pareto).
    Parametri:
    - files: lista di percorsi CSV
    - output_png: percorso file PNG di output
    - zoom: se effettuare o meno lo zoom
    """
    # Colonne standard
    x_col = "Rate_DL_py_load_test_tflite" # TODO: rimetti MCU
    y_col = "mean_tot_latency_fast"
    env_col = "env_name"
    settings_col = "end_folder_Training_Size_dd_epochs"
    err_cols = ("Error_does_not_fit", "Error_model_in_ram")

    dfs = []
    for file in files:
        if not os.path.exists(file):
            print(f"[AVVISO] File non trovato: {file}")
            continue
        df = pd.read_csv(file)
        if not set([x_col, y_col, env_col, settings_col]).issubset(df.columns):
            print(f"[AVVISO] Colonne mancanti in {file}")
            continue
        df = df[(df[err_cols[0]] == 0) & (df[err_cols[1]] == 0)]
        df = df.dropna(subset=[x_col, y_col])
        dfs.append(df)

    if not dfs:
        print("Nessun file CSV valido trovato.")
        return

    data_all = pd.concat(dfs, ignore_index=True)

    # -----------------------------
    # Normalizzazione nomi micro e rilevazione modelinram
    # -----------------------------
    # Per unificare eventuali env_name simili con suffisso "modelinram"
    #def normalize_micro(name: str) -> str:
    #    return str(name).replace("-modelinram", "").replace("_modelinram", "")
    def normalize_micro(name: str) -> str:
        s = str(name).strip()                          # rimuove spazi ai bordi
        s = re.sub(r'[-_]?modelinram$', '', s, flags=re.IGNORECASE)  # rimuove suffisso con o senza -/_
        s = re.sub(r'\s+', ' ', s)                     # comprime spazi interni multipli
        return s.lower()                               # normalizza case


    # Colonna micro_base senza suffisso
    data_all["micro_base"] = data_all[env_col].astype(str).apply(normalize_micro)

    # Colonna binaria: 1 se "modelinram" presente, 0 altrimenti
    #data_all["modelinram"] = data_all[env_col].str.contains("modelinram", case=False).astype(int)
    data_all["modelinram"] = data_all[env_col].astype(str).str.strip().str.lower().str.contains("modelinram").astype(int)


    # Zoom: filtriamo solo i punti vicini al fronte Pareto
    pf_global = pareto_front(data_all, x_col, y_col)

    if zoom:
        # Limiti intorno al fronte
        pad_x = (pf_global[x_col].max() - pf_global[x_col].min()) * 0.2
        pad_y = (pf_global[y_col].max() - pf_global[y_col].min()) * 0.2
        data_all = data_all[
            (data_all[x_col] >= pf_global[x_col].min() - pad_x) &
            (data_all[x_col] <= pf_global[x_col].max() + pad_x) &
            (data_all[y_col] >= pf_global[y_col].min() - pad_y) &
            (data_all[y_col] <= pf_global[y_col].max() + pad_y)
        ]

    # Micro e settings unici
    unique_micro = sorted(data_all["micro_base"].unique())
    unique_settings = sorted(data_all[settings_col].astype(str).unique())

    # DEBUG
    #print(sorted(repr(x) for x in data_all["micro_base"].unique()))

    # -----------------------------
    # Assegnazione simboli e colori
    # -----------------------------
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>', 'h', 'H']
    micro_to_marker = {m: markers[i % len(markers)] for i, m in enumerate(unique_micro)}

    # Colori tab20 per settings
    cmap = plt.get_cmap("tab10")
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

            pf_local = pareto_front(df_sm, x_col, y_col)
            if not pf_local.empty:
                ax.plot(pf_local[x_col], pf_local[y_col]/1000,
                        color=settings_to_color[s],
                        linestyle='--',
                        linewidth=1.0,
                        alpha=0.7,
                        label=f"Pareto Front ({m})")
                
            # Punti "modelinram" = simbolo pieno
            df_full = df_sm[df_sm["modelinram"] == 1]
            if not df_full.empty:
                ax.scatter(df_full[x_col], df_full[y_col]/1000,  # DIVISO PER 1000
                           s=scatter_point_size, marker=micro_to_marker[m],
                           facecolor=settings_to_color[s],
                           edgecolor="k", linewidths=0.7, alpha=0.95)
                
            df_hollow = df_sm[df_sm["modelinram"] == 0]
            if not df_hollow.empty:
                ax.scatter(df_hollow[x_col], df_hollow[y_col]/1000,  # DIVISO PER 1000
                           s=scatter_point_size, marker=micro_to_marker[m],
                           facecolor="none", edgecolor=settings_to_color[s],
                           linewidths=1.2, alpha=0.95)

    # -----------------------------
    # Fronte Pareto
    # -----------------------------
    pf_y_scaled = pf_global[y_col]/1000
    ax.plot(pf_global[x_col], pf_y_scaled, color="r", linestyle="-", linewidth=2.0, alpha=0.3)

    # Assi e titolo
    #ax.set_xlabel(x_col)
    ax.set_xlabel("Predicted Achievable Rate [bps/Hz]")
    #ax.set_ylabel(f"{y_col} (x1000)")
    ax.set_ylabel(f"Total Prediction Latency [ms]")
    ax.set_title("Pareto Plot: Rate vs Latency, Test Set Measurements (Zoom)" if zoom else "Pareto Plot: Rate vs Latency, Test Set Measurements")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Inverti asse X (rate alti a sinistra, ottimo in basso a sinistra)
    plt.gca().invert_xaxis()

    # -----------------------------
    # Legende multiple
    # -----------------------------
    # Simboli micro
    micro_legend = [
        Line2D([0], [0], marker=micro_to_marker[m], color='w', markerfacecolor='gray',
               markeredgecolor='k', linewidth=0, markersize=markersize, label=str(m))
        for m in unique_micro
    ]
    micro_legend.append(Line2D([0], [0], color='r', linewidth=1.2, label="Pareto Front (Global)", alpha=0.5))
    micro_legend.append(Line2D([0], [0], color='k', linewidth=1.2, label="Pareto Front (Local)", linestyle='--', ))
    leg1 = ax.legend(handles=micro_legend, title=f"{'MCU Type'.center(len(env_col)+20)}", loc="upper right", frameon=True)
    ax.add_artist(leg1)

    # Colori settings con line break per nomi lunghi
    settings_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=settings_to_color[s],
               markeredgecolor='k', linewidth=0, markersize=markersize, label="\n".join(textwrap.wrap(str(s), 20)))
        for s in unique_settings
    ]
    #leg2 = ax.legend(handles=settings_legend, title=f"{'Settings'.center(len(f'({settings_col})')+20)}\n({settings_col})", loc="upper right", frameon=True)
    leg2 = ax.legend(handles=settings_legend, title=f"{'Settings'.center(len(f'({settings_col})'))}", loc="upper center", frameon=True)
    ax.add_artist(leg2)

    # Legenda per variante micro e fronte Pareto
    fill_legend = [
        Line2D([0], [0], marker='o', color='k', markerfacecolor='k', markeredgecolor='k',
               linewidth=0, markersize=markersize, label="From RAM"),
        Line2D([0], [0], marker='o', color='k', markerfacecolor='none', markeredgecolor='k',
               linewidth=0, markersize=markersize, label="From Flash")
    ]
    ax.legend(handles=fill_legend, title="Model's Load Position", loc="center right", frameon=True)

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

    # Plot completo
    plot_pareto_scatter(files, output_figure, zoom=False)
    # Plot zoom sul fronte Pareto
    #plot_pareto_scatter(files, output_figure_zoom, zoom=True)
