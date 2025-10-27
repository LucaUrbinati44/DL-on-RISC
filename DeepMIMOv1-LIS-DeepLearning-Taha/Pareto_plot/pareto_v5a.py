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
My_ar = [32]
Mz_ar = [32]
Training_Size = [30000]
Training_Size_dd = Training_Size[0]
max_epochs = 200
active_cells = [1, 4, 8, 12, 28]
mcu_type_name_list = ['pico', 'nucleo-f446ze', 'esp32-s2-saola-tflm', 'nucleo-h753zi']

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
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes: for legend title
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=TINY_SIZE)    # legend fontsize
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Dimensione dei punti scatter
markersize = 10
scatter_point_size = markersize**2

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
            front_x.append(x + xoffset/x)
            #front_y.append(y - yoffset)
            #front_x.append(x*(1+xoffset))
            front_y.append(y*(1-yoffset))
            best_y = y

    return pd.DataFrame({x_col: front_x, y_col: front_y})


# -----------------------------
# Funzione per generare scatter plot
# -----------------------------
def plot_pareto_scatter(files, pareto_plot_folder, xlim=[0, 0], ylim=[0, 0], zoom=False, subopt=True, plot_modelinram=True, logscale=True):
    """
    Genera uno scatter plot Pareto a partire da una lista di CSV (completo o zoom sul fronte Pareto).
    Parametri:
    - files: lista di percorsi CSV
    - pareto_plot_folder: cartella di output
    - zoom: se effettuare o meno lo zoom
    """
    print(xlim)
    print(ylim)
    # Colonne standard
    x_col = "Rate_DL_py_load_test_tflite_mcu" # TODO: qui usare _mcu
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

    # Creazione del dataframe
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

    # Se plot_modelinram == False, rimuovi tutti i punti modelinram
    if plot_modelinram == False:
        data_all = data_all[data_all["modelinram"] == 0]

    # Zoom: filtriamo solo i punti vicini al fronte Pareto
    #pf_global = pareto_front(data_all, x_col, y_col, 0.001, 50)
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
    # Ordina unique_micro in base all'ordine di mcu_type_name_list
    def micro_order(micro):
        try:
            return mcu_type_name_list.index(micro)
        except ValueError:
            return len(mcu_type_name_list)
    #unique_micro = sorted(data_all["micro_base"].unique())
    unique_micro = sorted(data_all["micro_base"].unique(), key=micro_order)

    # Ordina unique_settings in base al numero dopo 'Mbar'
    def extract_mbar(s):
        match = re.search(r'Mbar(\d+)', str(s))
        return int(match.group(1)) if match else float('inf')
    #unique_settings = sorted(data_all[settings_col].astype(str).unique())
    unique_settings = sorted(data_all[settings_col].astype(str).unique(), key=extract_mbar)

    # DEBUG
    #print(sorted(repr(x) for x in data_all["micro_base"].unique()))

    # -----------------------------
    # Assegnazione simboli e colori
    # -----------------------------
    # https://matplotlib.org/stable/api/markers_api.html
    markers = ['^', '*', 's', 'o'] # 'd', 'D', 'v', 'p', 'P', 'X', '*', '<', '>', 'h', 'H'
    micro_to_marker = {m: markers[i % len(markers)] for i, m in enumerate(unique_micro)}

    # Colori tab20 per settings
    #cmap = plt.get_cmap("tab10")
    #settings_to_color = {s: cmap(i % cmap.N) for i, s in enumerate(unique_settings)}
    colors_list = ['tab:purple', 'tab:green', 'tab:red', 'tab:blue', 'tab:orange']
    settings_to_color = {s: colors_list[i] for i, s in enumerate(unique_settings)}


    # Creazione figura
    fig, ax = plt.subplots(figsize=(9, 7)) # default: 9, 7

    # -----------------------------
    # Plot dei punti scatter
    # -----------------------------
    # Se subopt == False, mantieni solo i punti appartenenti al fronte di Pareto (globale o locale)
    if subopt == False:
        # 1. Punti sul fronte Pareto globale
        pareto_points = set(zip(pf_global[x_col], pf_global[y_col]))

        # 2. Punti sui fronti locali (per ciascuna combinazione di micro_base e settings)
        for s in data_all[settings_col].unique():
            for m in data_all["micro_base"].unique():
                df_sm = data_all[(data_all[settings_col] == s) & (data_all["micro_base"] == m)]
                if df_sm.empty:
                    continue
                pf_local = pareto_front(df_sm, x_col, y_col)
                pareto_points.update(zip(pf_local[x_col], pf_local[y_col]))

        # 3. Filtra data_all mantenendo solo i punti appartenenti ai fronti (globali o locali)
        data_all = data_all[data_all.apply(lambda r: (r[x_col], r[y_col]) in pareto_points, axis=1)]

    for s in unique_settings:
        df_s = data_all[data_all[settings_col].astype(str) == s]
        for m in unique_micro:
            df_sm = df_s[df_s["micro_base"] == m]
            if df_sm.empty:
                continue

            pf_local = pareto_front(df_sm, x_col, y_col)
            if pf_local.empty:
                continue
            
            if not pf_local.empty:
                ax.plot(pf_local[x_col], pf_local[y_col]/1000,
                        color=settings_to_color[s],
                        linestyle='--',
                        linewidth=1.0,
                        alpha=0.75,
                        label=f"Pareto Front ({m})")
            
            # Punti senza "modelinram" = simbolo vuoto
            df_hollow = df_sm[df_sm["modelinram"] == 0]
            if not df_hollow.empty:
                ax.scatter(df_hollow[x_col], df_hollow[y_col]/1000,  # DIVISO PER 1000
                        s=scatter_point_size, marker=micro_to_marker[m],
                        facecolor="none", edgecolor=settings_to_color[s],
                        linewidths=1.2, alpha=0.99)
                
            if plot_modelinram == True:
                # Punti "modelinram" = simbolo pieno
                df_full = df_sm[df_sm["modelinram"] == 1]
                if not df_full.empty:
                    ax.scatter(df_full[x_col], df_full[y_col]/1000,  # DIVISO PER 1000
                            s=scatter_point_size, marker=micro_to_marker[m],
                            facecolor=settings_to_color[s],
                            edgecolor="k", linewidths=0.7, alpha=0.6)
            

    # -----------------------------
    # Linee verticali
    # -----------------------------
    # Recupera il valore della colonna 'Rate_OPT_py_load_test' nella prima riga che contiene 'M3232' in 'end_folder_Training_Size_dd_epochs'
    mask_M3232 = data_all["end_folder_Training_Size_dd_epochs"].astype(str).str.contains("M3232")
    # Recupera il valore della colonna 'Rate_OPT_py_load_test' nella prima riga che contiene 'M6464' in 'end_folder_Training_Size_dd_epochs'
    mask_M6464 = data_all["end_folder_Training_Size_dd_epochs"].astype(str).str.contains("M6464")
        
    if mask_M3232.any():
        if "Rate_OPT_py_load_test" in data_all.columns:
            rate_opt_M3232 = data_all.loc[mask_M3232, "Rate_OPT_py_load_test"].iloc[0]
            ax.axvline(rate_opt_M3232, color='b', linestyle=':', linewidth=1.2, label=f'{round(rate_opt_M3232,3)}                                                                                                                                                                ')
            labelLines([plt.gca().get_lines()[-1]], color='b', rotation=90, fontsize=SMALL_SIZE)
            line_rate_opt_M3232 = ax.axvline(rate_opt_M3232, color='b', linestyle=':', linewidth=1.2, label='Genie-Aided M=32x32')
        
    if mask_M6464.any():
        rate_opt_M6464 = data_all.loc[mask_M6464, "Rate_OPT_py_load_test"].iloc[0]
        ax.axvline(rate_opt_M6464, color='r', linestyle=':', linewidth=1.2, label=f'{round(rate_opt_M6464,3)}                                                                                                                                                                ')
        labelLines([plt.gca().get_lines()[-1]], color='r', rotation=90, fontsize=SMALL_SIZE)
        line_rate_opt_M6464 = ax.axvline(rate_opt_M6464, color='r', linestyle=':', linewidth=1.2, label='Genie-Aided M=64x64')

        #Parameters
        #----------
        #line : matplotlib.lines.Line
        #   The line holding the label
        #x : number
        #   The location in data unit of the label
        #label : string, optional
        #   The label to set. This is inferred from the line by default
        #align : boolean, optional
        #   If True, the label will be aligned with the slope of the line
        #   at the location of the label. If False, they will be horizontal.
        #drop_label : bool, optional
        #   If True, the label is consumed by the function so that subsequent
        #   calls to e.g. legend do not use it anymore.
        #xoffset : double, optional
        #    Space to add to label's x position
        #xoffset_logspace : bool, optional
        #    If True, then xoffset will be added to the label's x position in
        #    log10 space
        #yoffset : double, optional
        #    Space to add to label's y position
        #yoffset_logspace : bool, optional
        #    If True, then yoffset will be added to the label's y position in
        #    log10 space
        #outline_color : None | "auto" | color
        #    Colour of the outline. If set to "auto", use the background color.
        #    If set to None, do not draw an outline.
        #outline_width : number
        #    Width of the outline
        #rotation: float, optional
        #        If set and align = False, controls the angle of the label

    # -----------------------------
    # Fronte Pareto
    # -----------------------------
    pf_y_scaled = pf_global[y_col]/1000
    #ax.plot(pf_global[x_col], pf_y_scaled, color="r", linestyle="-", linewidth=1.5, alpha=0.5)
    ax.plot(pf_global[x_col], pf_y_scaled, color="gold", linestyle="-", linewidth=6, alpha=0.5)

    # -----------------------------
    # Assi e titolo
    # -----------------------------
    #ax.set_xlabel(x_col)
    ax.set_xlabel("Predicted Achievable Rate [bps/Hz]")
    #ax.set_ylabel(f"{y_col} (x1000)")
    ax.set_ylabel(f"Total Prediction Latency [ms]")
    ax.set_title(f"Pareto Plot: Rate vs Latency, Test Set\ncut{xlim}{ylim}_zoom{zoom}_subopt{subopt}_modelinram{plot_modelinram}")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Inverti asse X (rate alti a sinistra, ottimo in basso a sinistra)
    plt.gca().invert_xaxis()

    # -----------------------------
    # Legende multiple
    # -----------------------------
    # Simboli micro
    micro_legend = [
        Line2D([0], [0], marker=micro_to_marker[m], color='w', markerfacecolor='gray', markeredgecolor='k', linewidth=0, markersize=markersize, label=str(m))
        for m in unique_micro
    ]
    micro_legend.append(Line2D([0], [0], color='grey', linewidth=1.2, label="Pareto Front (Local)", linestyle='--'))
    #micro_legend.append(Line2D([0], [0], color='r', linewidth=1.2, label="Pareto Front (Global)", alpha=0.5))
    micro_legend.append(Line2D([0], [0], color='gold', linewidth=6, label="Pareto Front (Global)", alpha=0.8))
    micro_legend.append(Line2D([0], [0], color='w', label="\nModel's Load Position"))
    micro_legend.append(mpatches.Patch(edgecolor='k', facecolor='grey', linewidth=1, label="From RAM"))
    micro_legend.append(mpatches.Patch(edgecolor='grey', facecolor='none', linewidth=1, label="From Flash"))
    leg1 = ax.legend(handles=micro_legend, title=f"{'MCU Type'.center(len(env_col)+20)}", loc="upper right", frameon=True)
    ax.add_artist(leg1)

    # Colori settings con line break per nomi lunghi
    def format_settings_label(s):
        s = str(s)
        m_match = re.search(r'_M(\d{2})(\d{2})', s)
        mbar_match = re.search(r'Mbar(\d+)', s)
        m_str = f"M={m_match.group(1)}x{m_match.group(2)}" if m_match else ""
        mbar_str = r'$\overline{M}$' + f"={mbar_match.group(1)}" if mbar_match else ""
        return f"{m_str}, {mbar_str}".strip()
    settings_legend = [
        #Line2D([0], [0], color='w', marker='*', markerfacecolor=settings_to_color[s], linewidth=0, markersize=markersize+10, label="\n".join(textwrap.wrap(str(s), 20)))
        #mpatches.Patch(color=settings_to_color[s], linewidth=1, label="\n".join(textwrap.wrap(str(s), 20)))
        mpatches.Patch(color=settings_to_color[s], linewidth=1, label=format_settings_label(s))
        for s in unique_settings
    ]
    #settings_legend.append(Line2D([0], [0], color=settings_to_color[s], linestyle=':', linewidth=1.2, markersize=markersize, label="Genie-aided"))
    if mask_M3232.any():
        settings_legend.append(line_rate_opt_M3232)
    if mask_M6464.any():
        settings_legend.append(line_rate_opt_M6464)

    #leg2 = ax.legend(handles=settings_legend, title=f"{'Settings'.center(len(f'({settings_col})')+20)}\n({settings_col})", loc="upper right", frameon=True)
    #settings_legend.append(Line2D([0], [0], color='k', linestyle=':', linewidth=1.2, markersize=markersize, label="Genie-aided"))
    leg2 = ax.legend(handles=settings_legend, title=f"{'Settings'.center(len(f'({settings_col})'))}", loc="upper center", frameon=True)
    ax.add_artist(leg2)

    # Legenda per variante micro e fronte Pareto
    #fill_legend = [
    #    #Line2D([0], [0], marker='o', color='k', markerfacecolor='k', markeredgecolor='k', linewidth=0, markersize=markersize, label="From RAM"),
    #    #Line2D([0], [0], marker='o', color='k', markerfacecolor='none', markeredgecolor='k', linewidth=0, markersize=markersize, label="From Flash")
    #    mpatches.Patch(edgecolor='k', facecolor='gray', linewidth=1, label="From RAM"),
    #    mpatches.Patch(edgecolor='k', facecolor='none', linewidth=1, label="From Flash")
    #]
    #ax.legend(handles=fill_legend, title="Model's Load Position", loc="center right", frameon=True)

    # -----------------------------
    # Limiti e ticks
    # -----------------------------
    if xlim != [0, 0]:
        ax.set_xlim(xlim)
        plt.xticks(np.arange(xlim[1], xlim[0], step=0.01))
    if ylim != [0, 0]:
        ax.set_ylim(ylim)

    # -----------------------------
    # Log scale
    # -----------------------------
    if logscale == True:
        #ax.set_xscale('log')
        ax.set_yscale('log')

    # -----------------------------
    # Salvataggio figura
    # -----------------------------
    plt.tight_layout()

    output_png = os.path.join(pareto_plot_folder, f"pareto_plot{end_folder_Training_Size_dd_epochs}_cut{xlim}{ylim}_zoom{zoom}_subopt{subopt}_modelinram{plot_modelinram}_logscale{logscale}.png")
    plt.savefig(output_png, dpi=300)
    #plt.show()
    print(f"Salvato grafico in {output_png}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    # Plot completo
    #plot_pareto_scatter(files, pareto_plot_folder, zoom=False, subopt=True, plot_modelinram=True, logscale=True)
    #plot_pareto_scatter(files, pareto_plot_folder, zoom=False, subopt=False, plot_modelinram=True, logscale=True)
    
    # Plot tagliato
    xlim = [1.78, 1.35]
    ylim = [0.4, 450]
    #plot_pareto_scatter(files, pareto_plot_folder, xlim, ylim, zoom=False, subopt=False, plot_modelinram=True, logscale=True)
    
    xlim = [1.776, 1.64]
    ylim = [0.8, 450]
    plot_pareto_scatter(files, pareto_plot_folder, xlim, ylim, zoom=False, subopt=False, plot_modelinram=True, logscale=True)
    
    # Plot zoom sul fronte Pareto
    #plot_pareto_scatter(files, pareto_plot_folder, zoom=True, subopt=False, plot_modelinram=True, logscale=True)
    #plot_pareto_scatter(files, pareto_plot_folder, zoom=True, subopt=True, plot_modelinram=False, logscale=True)
    #plot_pareto_scatter(files, pareto_plot_folder, zoom=True, subopt=False, plot_modelinram=False, logscale=True)
