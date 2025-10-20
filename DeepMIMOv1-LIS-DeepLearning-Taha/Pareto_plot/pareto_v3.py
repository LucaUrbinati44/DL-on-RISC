#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -----------------------------
# Pareto front (maximize x, minimize y)
# -----------------------------
def pareto_front(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """
    Restituisce i vertici del fronte di Pareto globale con:
    - x (rate) da massimizzare
    - y (latenza) da minimizzare

    Procedura robusta:
    1) Aggrega per x tenendo la minima y (evita duplicati a pari rate).
    2) Ordina per x decrescente.
    3) Scansiona mantenendo y strettamente decrescente (miglioramento).
    """
    d = df[[x_col, y_col]].dropna().copy()
    # Step 1: per ogni x, tieni y minima
    d = d.groupby(x_col, as_index=False)[y_col].min()
    # Step 2: ordina per x decrescente
    d = d.sort_values(by=[x_col], ascending=False)

    # Step 3: scan monotona
    front_x, front_y = [], []
    best_y = np.inf
    for _, row in d.iterrows():
        x, y = row[x_col], row[y_col]
        if y < best_y:
            best_y = y
            front_x.append(x)
            front_y.append(y)

    return pd.DataFrame({x_col: front_x, y_col: front_y})

# -----------------------------
# Plot
# -----------------------------
def plot_pareto(csv_files: List[str],
                output_path: str,
                x_col: str = "Rate_DL_py_load_test_tflite_mcu",
                y_col: str = "mean_tot_latency",
                env_col: str = "env_name",
                settings_col: str = "end_folder_Training_Size_dd_epochs",
                err_cols: Tuple[str, str] = ("Error_does_not_fit", "Error_model_in_ram"),
                modelinram_col: str = "modelinram",
                title: str = "Pareto: Rate vs Latenza",
                figsize: Tuple[int, int] = (10, 7),
                dpi: int = 200) -> None:

    frames = []
    file_origin_col = "__file_origin__"
    for f in csv_files:
        if not os.path.isfile(f):
            print(f"ATTENZIONE: file non trovato: {f}")
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"Errore nel leggere {f}: {e}")
            continue

        required = {x_col, y_col, env_col, settings_col}.union(err_cols)
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"ATTENZIONE: nel file {f} mancano colonne: {missing}")
            continue

        dfe = df.copy()
        # Filtra righe errate
        for ec in err_cols:
            dfe[ec] = pd.to_numeric(dfe[ec], errors="coerce").fillna(0.0)
        dfe = dfe[(dfe[err_cols[0]] == 0) & (dfe[err_cols[1]] == 0)]

        # Tipi e NaN
        dfe = dfe.replace([np.inf, -np.inf], np.nan)
        dfe = dfe.dropna(subset=[x_col, y_col, env_col, settings_col])
        dfe[x_col] = pd.to_numeric(dfe[x_col], errors="coerce")
        dfe[y_col] = pd.to_numeric(dfe[y_col], errors="coerce")
        dfe = dfe.dropna(subset=[x_col, y_col])

        # Colonna modelinram: interpreta 1/0/True/False
        if modelinram_col in dfe.columns:
            # normalizza a {0,1}
            vals = dfe[modelinram_col].astype(str).str.lower().str.strip()
            dfe[modelinram_col] = vals.isin(["1", "true", "yes", "y", "t"]).astype(int)
        else:
            dfe[modelinram_col] = 0  # default: non in RAM

        dfe[file_origin_col] = os.path.basename(f)
        frames.append(dfe)

    if not frames:
        print("Nessun dato valido trovato nei CSV forniti.")
        return

    data_all = pd.concat(frames, ignore_index=True)

    # Micro base: unifica eventuali due env_name dello stesso micro
    # Assunzione: il nome del micro è la parte prima di eventuale suffisso legato a modelinram.
    # Se c'è una convenzione diversa, si può passare una funzione di normalizzazione via parametro.
    def normalize_micro(name: str) -> str:
        # Esempio: "nucleo-h753zi" e "nucleo-h753zi-modelinram" -> "nucleo-h753zi"
        return str(name).replace("-modelinram", "").replace("_modelinram", "")

    data_all["micro_base"] = data_all[env_col].astype(str).apply(normalize_micro)

    # Mappa simboli per micro_base
    unique_micro = sorted(data_all["micro_base"].unique())
    markers_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>', 'h', 'H', 'd', 'p']
    micro_to_marker: Dict[str, str] = {m: markers_cycle[i % len(markers_cycle)] for i, m in enumerate(unique_micro)}

    # Colori per settings
    unique_settings = sorted(data_all[settings_col].astype(str).unique())
    cmap = plt.get_cmap("tab20")
    settings_to_color: Dict[str, tuple] = {s: cmap(i % cmap.N) for i, s in enumerate(unique_settings)}

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter: per ogni settings e micro_base, plotta due categorie: modelinram==1 (pieno), modelinram==0 (cavo)
    for s in unique_settings:
        df_s = data_all[data_all[settings_col].astype(str) == s]
        for m in unique_micro:
            df_sm = df_s[df_s["micro_base"] == m]
            if df_sm.empty:
                continue

            # Pieno: modelinram==1
            df_full = df_sm[df_sm[modelinram_col] == 1]
            if not df_full.empty:
                ax.scatter(
                    df_full[x_col],
                    df_full[y_col],
                    s=54,
                    marker=micro_to_marker[m],
                    facecolor=settings_to_color[s],
                    edgecolor="k",
                    linewidths=0.7,
                    alpha=0.95,
                    label=None
                )

            # Cavo: modelinram==0 -> solo contorno colorato
            df_hollow = df_sm[df_sm[modelinram_col] == 0]
            if not df_hollow.empty:
                ax.scatter(
                    df_hollow[x_col],
                    df_hollow[y_col],
                    s=54,
                    marker=micro_to_marker[m],
                    facecolor="none",
                    edgecolor=settings_to_color[s],
                    linewidths=1.2,
                    alpha=0.95,
                    label=None
                )

    # Fronte di Pareto globale corretto
    pf = pareto_front(data_all, x_col=x_col, y_col=y_col)
    if not pf.empty:
        ax.plot(pf[x_col], pf[y_col], color="black", linestyle="-", linewidth=2.0, label="Fronte Pareto")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Legenda micro (simboli)
    micro_legend = [
        Line2D([0], [0], marker=micro_to_marker[m], color='w', markerfacecolor='gray',
               markeredgecolor='k', linewidth=0, markersize=8, label=str(m))
        for m in unique_micro
    ]
    leg1 = ax.legend(handles=micro_legend, title=f"Micro ({env_col})", loc="upper right", frameon=True)
    ax.add_artist(leg1)

    # Legenda settings (colori)
    settings_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=settings_to_color[s],
               markeredgecolor='k', linewidth=0, markersize=8, label=str(s))
        for s in unique_settings
    ]
    leg2 = ax.legend(handles=settings_legend, title=f"Settings ({settings_col})", loc="lower left", frameon=True)
    ax.add_artist(leg2)

    # Legenda riempimento (pieno vs cavo)
    fill_legend = [
        Line2D([0], [0], marker='o', color='k', markerfacecolor='k', markeredgecolor='k',
               linewidth=0, markersize=7, label="modelinram"),
        Line2D([0], [0], marker='o', color='k', markerfacecolor='none', markeredgecolor='k',
               linewidth=1.2, markersize=7, label="no modelinram"),
    ]
    ax.legend(handles=fill_legend, title="Variante micro", loc="center right", frameon=True)

    # Limiti con padding
    def pad_limits(vals, pad_ratio=0.05):
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return None
        span = vmax - vmin if vmax > vmin else max(abs(vmax), 1.0)
        pad = span * pad_ratio
        return (vmin - pad, vmax + pad)

    xlim = pad_limits(data_all[x_col].values)
    ylim = pad_limits(data_all[y_col].values)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    #plt.close(fig)
    plt.show()
    print(f"Salvato grafico complessivo in: {output_path}")

# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Plot Pareto Rate vs Latenza da CSV multipli (figura unica sovrapposta).")
    p.add_argument("csv", nargs='+', help="Lista di file CSV di input.")
    p.add_argument("--out", default="pareto_overall.png", help="Percorso file PNG di output.")
    p.add_argument("--title", default="Pareto: Rate vs Latenza", help="Titolo del grafico.")
    p.add_argument("--xcol", default="Rate_DL_py_load_test_tflite_mcu", help="Colonna asse X (rate).")
    p.add_argument("--ycol", default="mean_tot_latency", help="Colonna asse Y (latenza).")
    p.add_argument("--envcol", default="env_name", help="Colonna micro/env per simboli.")
    p.add_argument("--settingscol", default="end_folder_Training_Size_dd_epochs", help="Colonna settings per colori.")
    p.add_argument("--errcols", nargs=2, default=["Error_does_not_fit", "Error_model_in_ram"], help="Due colonne di errore (saltare se != 0).")
    p.add_argument("--modelinram", default="modelinram", help="Colonna booleana/int per variante in RAM.")
    p.add_argument("--dpi", type=int, default=200, help="DPI del PNG.")
    p.add_argument("--figw", type=int, default=10, help="Larghezza figura (pollici).")
    p.add_argument("--figh", type=int, default=7, help="Altezza figura (pollici).")
    return p.parse_args()

def main():
    # Esempio statico; scommentare parse_args per uso CLI
    # args = parse_args()
    # plot_pareto(
    #     csv_files=args.csv,
    #     output_path=args.out,
    #     x_col=args.xcol,
    #     y_col=args.ycol,
    #     env_col=args.envcol,
    #     settings_col=args.settingscol,
    #     err_cols=tuple(args.errcols),
    #     modelinram_col=args.modelinram,
    #     title=args.title,
    #     figsize=(args.figw, args.figh),
    #     dpi=args.dpi
    # )

    files = [
        'ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_nucleo-h753zi.csv',
        'ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_esp32-s2-saola-tflm.csv'
    ]
    output_png = 'pareto_plot.png'
    plot_pareto(
        csv_files=files,
        output_path=output_png,
        x_col="Rate_DL_py_load_test_tflite_mcu",
        y_col="mean_tot_latency",
        env_col="env_name",
        settings_col="end_folder_Training_Size_dd_epochs",
        err_cols=("Error_does_not_fit", "Error_model_in_ram"),
        modelinram_col="modelinram",
        title="Pareto: Rate vs Latenza",
        figsize=(10, 7),
        dpi=300
    )

if __name__ == "__main__":
    main()