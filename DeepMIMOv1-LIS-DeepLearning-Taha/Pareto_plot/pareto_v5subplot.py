#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pareto_plot_subplots.py

Genera una figura con subplot verticali (5x1) — ciascun subplot mostra i punti
(relative ad una singola "setting" identificata da 'MbarXX') per i casi 32x32 e 64x64.
Mantiene lo stile del tuo script originale, aggiungendo:
 - fronte Pareto globale del subplot (rosso, continuo)
 - fronti Pareto locali (per micro_base) tratteggiati, colorati come i punti
 - legenda unica per i "MCU Type" (in alto a destra della figura)
 - legenda settings centrata in alto di ogni subplot
 - opzioni per xlim/ylim per ogni subplot (passare dizionari)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os
import textwrap  # Per andare a capo nelle legende lunghe
import re
import subprocess
from labellines import labelLines

def is_windows():
    return 1 if os.name == 'nt' else 0
ISWINDOWS = is_windows()

base_folder = '/mnt/c/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/'
if ISWINDOWS:
    base_folder = subprocess.check_output(["wsl", "wslpath", "-w", base_folder]).decode().strip()
    print(base_folder)
output_folder = os.path.join(base_folder, 'Output_Python')
pareto_plot_folder = os.path.join(base_folder, 'Pareto_plot')
mcu_profiling_folder = os.path.join(output_folder, 'Profiling_Search_MCU')

debug = 0            # 0: production mode, 1: debug mode

seed = 0
K_DL = 64 # subcarriers, costante (per ora)
Ur_rows = [1000, 1200]
My_ar = [32, 64]
Mz_ar = [32, 64]
My_ar = [64]
My_ar = [32]
Mz_ar = My_ar
Training_Size = [30000]
Training_Size_dd = Training_Size[0]
max_epochs = 200
active_cells = [1, 4, 8, 12, 28]
mcu_type_name_list = ['pico', 'nucleo-f446ze', 'esp32-s2-saola-tflm', 'nucleo-h753zi']
mcu_type_name_lgd_list = ['RP2040', 'STM32-F446ZE', 'ESP32-S2-SOLO', 'STM32-H753ZI']

if debug == 1:
    files = [
        # Inserire i CSV qui
        os.path.join(pareto_plot_folder, "ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_esp32-s2-saola-tflm.csv"),
        #os.path.join(pareto_plot_folder, "ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_nucleo-h753zi.csv"),
        os.path.join(pareto_plot_folder, "ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_nucleo-h753zi_fake.csv"),
        os.path.join(pareto_plot_folder, "ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_fake_esp32-s2-saola-tflm.csv"),
        os.path.join(pareto_plot_folder, "ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_fake2_esp32-s2-saola-tflm.csv"),
    ]
else:
    files = [] # Inserire i CSV qui
    for My, Mz in zip(My_ar, Mz_ar):
        for M_bar in active_cells:
            for mcu_type_name in mcu_type_name_list:

                end_folder = '_seed' + str(seed) + '_grid' + str(Ur_rows[1]) + '_M' + str(My) + str(Mz) + '_Mbar' + str(M_bar)
                end_folder_Training_Size_dd = end_folder + '_' + str(Training_Size_dd)
                end_folder_Training_Size_dd_epochs = end_folder_Training_Size_dd + f"_ep{str(max_epochs)}"

                files.append(os.path.join(mcu_profiling_folder, f"profiling{end_folder_Training_Size_dd_epochs}_{mcu_type_name}.csv"))

# Imposta font size per ogni elemento del grafico
TINY_SIZE = 10
SMALL_SIZE = 12
MEDIUM_SIZE = 14
LARGE_SIZE = 16
XL_SIZE = 18
XXL_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes: for legend title
plt.rc('xtick', labelsize=TINY_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TINY_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=TINY_SIZE)    # legend fontsize
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('axes', titlesize=LARGE_SIZE)     # fontsize of the axes title
plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title

# Dimensione dei punti scatter
markersize = 12
scatter_point_size = markersize**2
LINEWIDTH_TINY = TINY_SIZE/10
LINEWIDTH_SMALL = SMALL_SIZE/10
LINEWIDTH_MEDIUM = MEDIUM_SIZE/10
LINEWIDTH_LARGE = LARGE_SIZE/10
LINEWIDTH_XL = XL_SIZE/10
LINEWIDTH_XXL = XXL_SIZE/10

# -----------------------------
# Funzione per calcolare il fronte di Pareto
# -----------------------------
def pareto_front(df, x_col, y_col, xoffset=0.0, yoffset=0.0):
    """
    Calcola il fronte di Pareto (massimizza x, minimizza y).
    Parametri:
    - df: DataFrame contenente tutti i punti
    - x_col: nome colonna asse X (rate)
    - y_col: nome colonna asse Y (latenza)
    - xoffset: offset in percentuale es. 0.01 = 1% per non far sovrapporre la riga rossa sulla tratteggiata locale
    - yoffset: idem
    Ritorna:
    - DataFrame dei punti del fronte Pareto (x_col, y_col)
    Procedura:
    1. Rimuove valori NaN
    2. Per ogni valore x mantiene solo y minima
    3. Ordina x decrescente e mantiene solo punti con y decrescente (miglioramento)
    """
    # Copia dei dati e rimozione NaN
    d = df[[x_col, y_col]].dropna().copy()
    if d.empty:
        return pd.DataFrame({x_col: [], y_col: []})
    
    # Step 1: per ogni x, tieni solo y minima
    d = d.groupby(x_col, as_index=False)[y_col].min()

    # Step 2: ordina per x decrescente (rate alto a sinistra)
    d = d.sort_values(by=[x_col], ascending=False)

    # Step 3: scansione per filtrare y decrescenti
    front_x, front_y = [], []
    best_x = 0
    best_y = np.inf
    for _, row in d.iterrows():
        x, y = row[x_col], row[y_col]
        #if y < best_y:
        #if y <= best_y and abs(x - best_x) > 0.001:
        if y < best_y*(1-0.01) and abs(x - best_x) > 0.001: # un punto è di pareto se supera il precedente dell'1% in y (latenza)
            front_x.append(x + xoffset/x)
            #front_y.append(y - yoffset)
            #front_x.append(x*(1+xoffset))
            front_y.append(y*(1-yoffset))
            best_x = x
            best_y = y
            
    return pd.DataFrame({x_col: front_x, y_col: front_y})

# -----------------------------
# Normalizzazione nomi micro e rilevazione modelinram
# -----------------------------
# Per unificare eventuali env_name simili con suffisso "modelinram"
def normalize_micro(name: str) -> str:
    """
    Normalizza env_name per ottenere il micro_base (rimuove suffisso modelinram,
    comprime spazi e porta in minuscolo).
    """
    s = str(name).strip()                          # rimuove spazi ai bordi
    s = re.sub(r'[-_]?modelinram$', '', s, flags=re.IGNORECASE)  # rimuove suffisso con o senza -/_
    s = re.sub(r'\s+', ' ', s)                     # comprime spazi interni multipli
    return s.lower()                               # normalizza case
    
def extract_mbar(s: str):
    """
    Estrae il valore numerico di Mbar da una stringa di settings.
    Ritorna int o inf se non trovato.
    """
    match = re.search(r'Mbar(\d+)', str(s))
    return int(match.group(1)) if match else float('inf')

# -----------------------------
# Funzione principale: subplot per settings (Mbar)
# -----------------------------
def plot_pareto_subplots(files,
                         pareto_plot_folder,
                         active_cells_list=None,
                         x_col="Rate_DL_py_load_test_tflite_mcu",
                         y_col="mean_tot_latency_fast",
                         env_col="env_name",
                         settings_col="end_folder_Training_Size_dd_epochs",
                         err_cols=("Error_does_not_fit", "Error_model_in_ram"),
                         plot_modelinram=True,
                         logscale=True,
                         axes_lims_enable=False,
                         xlims_dict=None,
                         ylims_dict=None,
                         zoom=False,
                         subopt=True,
                         export_pdf=False):
    """
    Crea una figura con N subplot verticali (N = len(active_cells_list)),
    uno per ciascun valore di Mbar (active_cells_list). Ogni subplot contiene
    solo le settings che includono quel Mbar e i casi 32x32/64x64 (se presenti).

    Parametri principali:
    - files: lista file CSV (uguale al tuo script precedente)
    - pareto_plot_folder: cartella dove salvare il PNG
    - active_cells_list: lista di Mbar (di default usa `active_cells` globale)
    - x_col/y_col/env_col/settings_col: nomi colonne CSV
    - plot_modelinram: se False rimuove i punti modelinram
    - logscale: applica scala log su y (come nel tuo script)
    - xlims_dict / ylims_dict: dizionari {mbar: (xmin,xmax) / (ymin,ymax)} per personalizzare ogni subplot
    - zoom: se True applica zoom centrato sul fronte Pareto del subplot
    - subopt: se False mantiene solo i punti appartenenti ai fronti (globale o locali)
    """
    if active_cells_list is None:
        active_cells_list = active_cells[:]  # default

    if axes_lims_enable:
        if xlims_dict is None:
            xlims_dict = {}
        if ylims_dict is None:
            ylims_dict = {}

    # --- carica e concatena CSV come nel tuo script ---
    dfs = []
    for f in files:
        if not os.path.exists(f):
            print(f"[AVVISO] File non trovato: {f}")
            continue
        df = pd.read_csv(f)
        required = {x_col, y_col, env_col, settings_col}.union(set(err_cols))
        if not required.issubset(df.columns):
            print(f"[AVVISO] Colonne mancanti in {f}: {required - set(df.columns)}")
            continue
        # filtra error rows come prima
        df = df[(df[err_cols[0]] == 0) & (df[err_cols[1]] == 0)]
        df = df.dropna(subset=[x_col, y_col])
        dfs.append(df)

    if not dfs:
        print("Nessun file CSV valido trovato.")
        return

    # Creazione del dataframe
    data_all = pd.concat(dfs, ignore_index=True)

    # Normalizzazione micro e detection modelinram
    data_all["micro_base"] = data_all[env_col].astype(str).apply(normalize_micro)

    # Colonna binaria: 1 se "modelinram" presente, 0 altrimenti
    data_all["modelinram"] = data_all[env_col].astype(str).str.strip().str.lower().str.contains("modelinram").astype(int)

    # Se plot_modelinram == False, rimuovi tutti i punti modelinram
    if plot_modelinram == False:
        data_all = data_all[data_all["modelinram"] == 0]

    # Ordina micro secondo mcu_type_name_list, altrimenti append in fondo
    def micro_order(micro):
        try:
            return mcu_type_name_list.index(micro)
        except ValueError:
            return len(mcu_type_name_list)
    unique_micro_global = sorted(data_all["micro_base"].unique(), key=micro_order)

    # -----------------------------
    # Assegnazione simboli e colori
    # -----------------------------
    # https://matplotlib.org/stable/api/markers_api.html
    markers = ['^', '*', 's', 'o'] # 'd', 'D', 'v', 'p', 'P', 'X', '*', '<', '>', 'h', 'H'
    micro_to_marker = {m: markers[i % len(markers)] for i, m in enumerate(unique_micro_global)}
    
    # Colori tab20 per settings
    colors_list = ['tab:purple', 'tab:green', 'tab:red', 'tab:blue', 'tab:orange']
    # user may have many distinct settings per Mbar; we will map them as encountered
    settings_unique_all = sorted(data_all[settings_col].astype(str).unique(), key=extract_mbar)
    settings_to_color = {s: colors_list[i % len(colors_list)] for i, s in enumerate(settings_unique_all)}

    # --- Preparazione figura con N subplot verticali (len(active_cells_list)) ---
    n_sub = len(active_cells_list)
    #fig, axes = plt.subplots(n_sub, 1, figsize=(7, 2.2 * n_sub), sharex=False)
    fig, axes = plt.subplots(n_sub, 1, figsize=(6, 3.2 * n_sub), sharex=False)
    fig.subplots_adjust(wspace=0, hspace=0)

    # se un solo subplot, axes non è array -> normalizziamo
    if n_sub == 1:
        axes = np.array([axes])

    # Per la legenda MCU unica costruisco gli handle (simboli)
    micro_legend = []
    for m, m_lgd in zip(unique_micro_global, mcu_type_name_lgd_list):
        micro_legend.append(Line2D([0], [0],
                                  marker=micro_to_marker[m],
                                  color='w',
                                  markerfacecolor='gray',
                                  markeredgecolor='k',
                                  markersize=markersize,
                                  linestyle='None',
                                  label=str(m_lgd)))
    # Handles per "Model's Load Position"
    load_legend = [
        #Line2D([0], [0], color='w', label="Model's Load Position"),
        mpatches.Patch(edgecolor='k', facecolor='grey', linewidth=LINEWIDTH_TINY, label="From RAM"),
        mpatches.Patch(edgecolor='grey', facecolor='none', linewidth=LINEWIDTH_TINY, label="From Flash"),
        Line2D([0], [0], color='w', label=""),
        Line2D([0], [0], color='w', label=""),
    ]

    pareto_legend = [
        Line2D([0], [0], color='grey', linewidth=LINEWIDTH_XXL, label="Local", linestyle='--', ),
        #Line2D([0], [0], color='r', linewidth=1.2, label="Global", alpha=0.5),
        #Line2D([0], [0], color='gold', linewidth=6, label="Global", alpha=0.8),
        Line2D([0], [0], color='w', label=""),
        Line2D([0], [0], color='w', label=""),
    ]

    # -----------------------------
    # Plot dei punti scatter
    # -----------------------------
    # Itero sui subplot in ordine di active_cells_list
    for ax, mbar in zip(axes, active_cells_list):
        # filtro le settings che corrispondono esattamente a "Mbar{mbar}"
        # uso regex per evitare che Mbar1 combaci con Mbar12
        pattern = f'_Mbar{mbar}_'
        # pandas str.contains con regex=True usa default na=False
        df_subplot = data_all[data_all[settings_col].astype(str).str.contains(pattern, regex=True, na=False)]
        if df_subplot.empty:
            # subplot vuoto: mantieni asse ma segna testo
            ax.text(0.5, 0.5, f"No data for Mbar{mbar}", ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue

        # Mantieni solo coppie 32x32 o 64x64 (controllo tramite substring M3232 o M6464)
        df_M3232 = df_subplot[df_subplot[settings_col].astype(str).str.contains("M3232")]
        df_M6464 = df_subplot[df_subplot[settings_col].astype(str).str.contains("M6464")]
        # Combina: se vuoi abilitare/disabilitare 64x64 basta che l'utente rimuova o aggiunga righe ai CSV
        df_subplot_filtered = pd.concat([df_M3232, df_M6464], ignore_index=True) if (not df_M3232.empty or not df_M6464.empty) else df_subplot

        # Calcola fronte Pareto globale del subplot
        #offset_x = 0.002
        #offset_y = 0.005
        offset_x = 0.00
        offset_y = 0.00
        pf_sub = pareto_front(df_subplot_filtered, x_col, y_col, offset_x, offset_y)

        # ---- se subopt == False: mantieni solo punti appartenenti ai fronti (global o locali) ----
        if subopt == False:
            pareto_points_set = set()

             # aggiungi metadati associando le coordinate ai record originali
            pf_sub = pf_sub.merge(
                df_subplot_filtered[[x_col, y_col, "micro_base", "modelinram"]],
                on=[x_col, y_col],
                how="left"
            )
            
            # punti frontali globali
            pareto_points_set.update(zip(pf_sub[x_col].tolist(),pf_sub[y_col].tolist(),pf_sub["micro_base"],pf_sub["modelinram"]))

            # fronti locali per ogni micro+setting
            for s in df_subplot_filtered[settings_col].astype(str).unique():
                df_s = df_subplot_filtered[df_subplot_filtered[settings_col].astype(str) == s]
                for m in df_s["micro_base"].unique():
                    df_sm = df_s[df_s["micro_base"] == m]
                    pf_local = pareto_front(df_sm, x_col, y_col)
                    if pf_local.empty:
                        continue
                    pf_local = pf_local.merge(df_sm[[x_col, y_col, "micro_base", "modelinram"]], on=[x_col, y_col], how="left")
                    pareto_points_set.update(zip(pf_local[x_col].tolist(),pf_local[y_col].tolist(),pf_local["micro_base"],pf_local["modelinram"]))

            # filtro mantenendo anche chi ha stessi (x,y) ma diversi modelinram, mantenendo quindi anche i punti sovrapposti
            df_subplot_filtered = df_subplot_filtered[df_subplot_filtered.apply(lambda r: (r[x_col], r[y_col], r["micro_base"], r["modelinram"]) in pareto_points_set,axis=1)]
            
            # ricalcolo pf_sub sulla sotto-porzione filtrata
            pf_sub = pareto_front(df_subplot_filtered, x_col, y_col, offset_x, offset_y)

        # Stampa solo le colonne utili per capire quali punti sono rimasti
        cols_to_show = [x_col, y_col, "micro_base", "modelinram", settings_col]
        #print(df_subplot_filtered[cols_to_show].sort_values(by=[x_col, y_col, "micro_base", "modelinram"]).to_string(index=False))
        extra_cols = ["input_features", "num_layers", "R", "hidden_units_list", "output_dim"]
        extra_cols2 = ["Rate_OPT_py_load_test", "Rate_DL_py_load_test", "Rate_DL_py_load_test_tflite"]
        # Aggiungi solo le colonne effettivamente presenti nel DataFrame
        cols_to_show += [c for c in extra_cols if c in df_subplot_filtered.columns]
        cols_to_show += [c for c in extra_cols2 if c in df_subplot_filtered.columns]
        # Stampa i punti ordinati con le colonne richieste
        print(df_subplot_filtered[cols_to_show]
            .sort_values(by=[x_col, y_col, "micro_base", "modelinram"])
            .to_string(index=False))
        
        # Per ciascuna settings presente nel subplot: plottiamo punti e fronte locale (per micro)
        settings_in_subplot = sorted(df_subplot_filtered[settings_col].astype(str).unique(), key=extract_mbar)
        print(settings_in_subplot)
        for s in settings_in_subplot:
            df_s = df_subplot_filtered[df_subplot_filtered[settings_col].astype(str) == s]
            # per ciascun micro base
            for m in sorted(df_s["micro_base"].unique(), key=micro_order):
                df_sm = df_s[df_s["micro_base"] == m]
                if df_sm.empty:
                    continue

                # fronte locale per questa coppia micro+setting
                pf_local = pareto_front(df_sm, x_col, y_col)
                if not pf_local.empty:
                    ax.plot(pf_local[x_col], pf_local[y_col] / 1000.0,  # y diviso 1000
                            color=settings_to_color.get(s, 'gray'),
                            linestyle='--',
                            linewidth=LINEWIDTH_MEDIUM,
                            alpha=0.75)

                # punti hollow (no modelinram)
                df_hollow = df_sm[df_sm["modelinram"] == 0]
                if not df_hollow.empty:
                    ax.scatter(df_hollow[x_col], df_hollow[y_col] / 1000.0,
                               s=scatter_point_size, marker=micro_to_marker[m],
                               facecolor='none', edgecolor=settings_to_color.get(s, 'gray'),
                               linewidths=LINEWIDTH_MEDIUM, alpha=0.95)

                # punti full (modelinram): plotta solo se abilitato
                if plot_modelinram:
                    df_full = df_sm[df_sm["modelinram"] == 1]
                    if not df_full.empty:
                        ax.scatter(df_full[x_col], df_full[y_col] / 1000.0,
                                   s=scatter_point_size, marker=micro_to_marker[m],
                                   facecolor=settings_to_color.get(s, 'gray'),
                                   edgecolor='k', linewidths=LINEWIDTH_TINY, alpha=0.6)

        # -----------------------------
        # Fronte Pareto (Pareto Front)
        # -----------------------------
        #if not pf_sub.empty:
        #    ax.plot(pf_sub[x_col], pf_sub[y_col] / 1000.0, color='gold', linestyle='-', linewidth=6, alpha=0.3, label='Pareto Front (Global)')

        # ---- se zoom == True: limita i dati del subplot intorno al fronte Pareto per centrarlo ----
        if zoom and (not pf_sub.empty):
            pad_x = (pf_sub[x_col].max() - pf_sub[x_col].min()) * 0.2 if pf_sub[x_col].max() != pf_sub[x_col].min() else 0.01
            pad_y = (pf_sub[y_col].max() - pf_sub[y_col].min()) * 0.2 if pf_sub[y_col].max() != pf_sub[y_col].min() else 1.0
            # ricampioniamo i punti visualizzati restringendo l'asse; mantenere solo punti nel riquadro
            df_mask_zoom = df_subplot_filtered[
                (df_subplot_filtered[x_col] >= pf_sub[x_col].min() - pad_x) &
                (df_subplot_filtered[x_col] <= pf_sub[x_col].max() + pad_x) &
                (df_subplot_filtered[y_col] >= pf_sub[y_col].min() - pad_y) &
                (df_subplot_filtered[y_col] <= pf_sub[y_col].max() + pad_y)
            ]
            # Se il filtro lascia qualcosa, ri-plottiamo solo quei punti; per semplicità, settiamo limiti sugli assi
            if not df_mask_zoom.empty:
                ax.set_xlim((pf_sub[x_col].max() + pad_x, pf_sub[x_col].min() - pad_x))  # invertito
                ax.set_ylim((max(1e-9, (pf_sub[y_col].min() - pad_y) / 1000.0), (pf_sub[y_col].max() + pad_y) / 1000.0))

        # -----------------------------
        # Linee verticali
        # -----------------------------
        # Genie-aided vertical lines (se presenti nel df_subplot_filtered per la M=32x32 o M=64x64)
        mask_M3232 = df_subplot_filtered[settings_col].astype(str).str.contains("M3232") & df_subplot_filtered[settings_col].astype(str).str.contains(pattern)
        mask_M6464 = df_subplot_filtered[settings_col].astype(str).str.contains("M6464") & df_subplot_filtered[settings_col].astype(str).str.contains(pattern)
        if mask_M3232.any() and "Rate_OPT_py_load_test" in df_subplot_filtered.columns:
            rate_opt_M3232 = df_subplot_filtered.loc[mask_M3232, "Rate_OPT_py_load_test"].iloc[0]
            rate_opt_M3232 = round(rate_opt_M3232,3)
            #ax.axvline(rate_opt_M3232, color='b', linestyle=':', linewidth=1.2, alpha=0.9, label=f'{round(rate_opt_M3232,3)}                                                                                                                                                                ')
            #labelLines(plt.gca().get_lines(), color='b', rotation=90, fontsize=SMALL_SIZE)
            line_rate_opt_M3232 = ax.axvline(rate_opt_M3232, color='b', linestyle=':', linewidth=LINEWIDTH_XL, label=f'Genie M=32x32: {round(rate_opt_M3232,3)} [bps/Hz]')
        if mask_M6464.any() and "Rate_OPT_py_load_test" in df_subplot_filtered.columns:
            rate_opt_M6464 = df_subplot_filtered.loc[mask_M6464, "Rate_OPT_py_load_test"].iloc[0]
            rate_opt_M6464 = round(rate_opt_M6464,3)
            #ax.axvline(rate_opt_M6464, color='r', linestyle=':', linewidth=1.2, alpha=0.9, label=f'{round(rate_opt_M6464,3)}                                                                                                                                                                ')
            #labelLines([plt.gca().get_lines()[-1]], color='r', rotation=90, fontsize=SMALL_SIZE)
            line_rate_opt_M6464 = ax.axvline(rate_opt_M6464, color='r', linestyle=':', linewidth=LINEWIDTH_XL, label=f'Genie M=64x64: {round(rate_opt_M6464,3)} [bps/Hz]')

        # -----------------------------
        # Assi e titolo
        # -----------------------------
        if mbar == active_cells_list[-1]: 
            ax.set_xlabel("Predicted Achievable Rate on the Test set [bps/Hz]")
        ax.set_ylabel("Total Prediction Latency [ms]")
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.invert_xaxis()  # rate alto a sinistra

        # -----------------------------
        # Log scale
        # -----------------------------
        if logscale:
            ax.set_yscale('log')

        # -----------------------------
        # Limiti e ticks
        # -----------------------------
        if axes_lims_enable:
            if mbar in xlims_dict:
                ax.set_xlim(xlims_dict[mbar])
            if mbar in ylims_dict:
                ax.set_ylim(ylims_dict[mbar])

        # Legenda settings centrata in alto del singolo subplot (mostra i colori delle settings presenti)
        settings_legend = []
        for s in settings_in_subplot:
            label_text = format_settings_label_verbose(s) if 'format_settings_label_verbose' in globals() else ("\n".join(textwrap.wrap(str(s), 30)))
            settings_legend.append(mpatches.Patch(color=settings_to_color.get(s, 'gray'), label=label_text))

        if settings_legend:
            if mask_M3232.any():
                settings_legend.append(line_rate_opt_M3232)
            if mask_M6464.any():
                settings_legend.append(line_rate_opt_M6464)
                
            # legenda centrata in alto del subplot
            #ax.legend(handles=settings_legend, title="", loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=min(len(settings_legend), 3), frameon=True)
            ax.legend(handles=settings_legend, title="", loc='upper right', frameon=True)

        # posiziona la legenda nel primo subplot
        #if mbar == 1: 
        #    leg2 = ax.legend(handles=micro_legend, title="MCU Type", loc='upper right', frameon=True)
        #    ax.add_artist(leg2)

    # -----------------------------
    # Legenda unica sotto il titolo della figura
    # -----------------------------
    # posizioniamo la legenda centrale subito sotto il suptitle, evitando spazio bianco tra suptitle e legenda
    #fig.suptitle("Pareto Plot: Rate vs Latency on the Test Set", y=0.95, fontsize=LARGE_SIZE)

    # Combiniamo handles per la legenda globale: prima i micro, poi i due patch (RAM/Flash)
    combined_handles = micro_legend + load_legend + pareto_legend

    # posizioniamo la legenda al centro, leggermente sotto il titolo (y ~ 0.96),
    # in modo che non lasci spazio bianco con il primo subplot.
    legend_box = fig.legend(handles=combined_handles,
                            title="    MCU Type             Model's Load Position     Pareto Front",
                            loc='upper center',
                            bbox_to_anchor=(0.54, 1.00), # the bbox (x, y, width, height)
                            ncol=3,
                            frameon=True,
                            #prop={'size': TINY_SIZE},
                            columnspacing=4.0)
    # riduco il padding tra suptitle e legenda
    #legend_box._legend_box.align = "left"

    # per spostare i subplot più in alto o più in basso
    #plt.tight_layout(rect=[0, 0, 1.0, 0.93])
    plt.tight_layout(pad=0.25, rect=[0, 0, 1.0, 0.925])

    # -----------------------------
    # Salvataggio figura
    # -----------------------------
    #output_fname = f"pareto_plot_subplots_{Training_Size_dd}_log{int(bool(logscale))}_zoom{int(bool(zoom))}_subopt{int(bool(subopt))}.png"  
    if export_pdf == False:
        output_png = os.path.join(pareto_plot_folder, f"pareto_subplots{end_folder_Training_Size_dd_epochs}_zoom{zoom}_subopt{subopt}_axeslims{axes_lims_enable}_modelinram{plot_modelinram}_logscale{logscale}.png")
    elif export_pdf == True: 
        output_png = os.path.join(pareto_plot_folder, f"pareto_subplots{end_folder_Training_Size_dd_epochs}_zoom{zoom}_subopt{subopt}_axeslims{axes_lims_enable}_modelinram{plot_modelinram}_logscale{logscale}.pdf")
    output_path = os.path.join(pareto_plot_folder, output_png)
    os.makedirs(pareto_plot_folder, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    print(f"Salvato subplot figure in: {output_path}")
    # non chiudiamo la figura: utente può plt.show() esternamente se desidera

# -----------------------------
# Helper (opzionale): format label settings (come usavi)
# -----------------------------
def format_settings_label_verbose(s):
    """
    Format breve per la label delle settings: estrae M=32x32 dal pattern _M3232 ecc.
    Restituisce una stringa compatta per le legende.
    """
    s = str(s)
    m_match = re.search(r'_M(\d{2})(\d{2})', s)
    mbar_match = re.search(r'Mbar(\d+)', s)
    m_str = f"M={m_match.group(1)}x{m_match.group(2)}" if m_match else ""
    mbar_str = r'$\overline{M}$' + f"={mbar_match.group(1)}" if mbar_match else ""
    joined = ", ".join([part for part in (m_str, mbar_str) if part])
    return joined if joined else s

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
   
    if My_ar[0] == 32:
        xlims = {
            1:  (0.35, 0.0),  # esempio (max left, min right) — ricorda che l'asse X viene invertito
            4:  (1.78, 1.35),
            8:  (1.78, 1.64),
            12: (1.78, 1.64),
            28: (1.78, 1.64),
        }
        ylims = {
            1:  (0.1, 500),
            4:  (0.1, 500),
            8:  (1, 500),
            12: (1, 500),
            28: (1, 500),
        }
    elif My_ar[0] == 64:
        xlims = {
            1:  (0.85, 0.0),  # esempio (max left, min right) — ricorda che l'asse X viene invertito
            4:  (5.25, 0.0),
            8:  (5.10, 3.8),
            12: (5.12, 3.8),
            28: (5.10, 3.8),
        }
        ylims = {
            1:  (0.1, 700),
            4:  (1,   700),
            8:  (1,   700),
            12: (1,   700),
            28: (1,   700),
        }

    # Chiamata principale: genera la figura con 5 subplot (active_cells)
    #plot_pareto_subplots(files,
    #                     pareto_plot_folder,
    #                     active_cells_list=active_cells,
    #                     x_col="Rate_DL_py_load_test_tflite_mcu",
    #                     y_col="mean_tot_latency_fast",
    #                     env_col="env_name",
    #                     settings_col="end_folder_Training_Size_dd_epochs",
    #                     err_cols=("Error_does_not_fit", "Error_model_in_ram"),
    #                     plot_modelinram=True,
    #                     logscale=True,
    #                     axes_lims_enable=False,
    #                     xlims_dict=xlims,
    #                     ylims_dict=ylims,
    #                     zoom=False,
    #                     subopt=True,
    #                     export_pdf=True)

    plot_pareto_subplots(files,
                         pareto_plot_folder,
                         active_cells_list=active_cells,
                         x_col="Rate_DL_py_load_test_tflite_mcu",
                         y_col="mean_tot_latency_fast",
                         env_col="env_name",
                         settings_col="end_folder_Training_Size_dd_epochs",
                         err_cols=("Error_does_not_fit", "Error_model_in_ram"),
                         plot_modelinram=True,
                         logscale=True,
                         axes_lims_enable=True,
                         xlims_dict=xlims,
                         ylims_dict=ylims,
                         zoom=False,
                         subopt=False, 
                         export_pdf=True)
