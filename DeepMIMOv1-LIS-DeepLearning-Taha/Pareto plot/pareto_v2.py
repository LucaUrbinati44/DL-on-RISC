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
# Utility: Pareto front (maximize rate, minimize latency)
# -----------------------------
def pareto_front(df: pd.DataFrame,
                 x_col: str,
                 y_col: str) -> pd.DataFrame:
    """
    Restituisce i punti sul fronte di Pareto considerando:
    - x (rate) da massimizzare
    - y (latenza) da minimizzare

    Strategia:
    1) Ordina per x decrescente; a parità di x, per y crescente.
    2) Scansiona e tieni i punti con y strettamente minore del minimo visto finora.
    """
    d = df[[x_col, y_col]].dropna().copy()
    d = d.sort_values(by=[x_col, y_col], ascending=[False, True])
    front = []
    best_y = np.inf
    for _, row in d.iterrows():
        y = row[y_col]
        if y < best_y:
            best_y = y
            front.append((row[x_col], row[y_col]))
    front_df = pd.DataFrame(front, columns=[x_col, y_col])
    return front_df

# -----------------------------
# Plot function
# -----------------------------
def plot_pareto(csv_files: List[str],
                output_path: str,
                x_col: str = "Rate_DL_py_load_test_tflite_mcu",
                y_col: str = "mean_tot_latency",
                env_col: str = "env_name",
                settings_col: str = "end_folder_Training_Size_dd_epochs",
                err_cols: Tuple[str, str] = ("Error_does_not_fit", "Error_model_in_ram"),
                title: str = "Pareto: Rate vs Latenza",
                figsize: Tuple[int, int] = (10, 7),
                dpi: int = 150) -> None:
    """
    - Itera su una lista di CSV.
    - Filtra righe con errori (err_cols != 0).
    - Scatter: simboli per env_name, colori per settings_col.
    - Fronte di Pareto (max x, min y) unito da spezzata nera.
    - Salva un PNG complessivo e uno per ciascun CSV se richiesto.
    """
    # Colleziona tutti i dati per il grafico complessivo
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

        # Controlli colonne indispensabili
        required = {x_col, y_col, env_col, settings_col}.union(err_cols)
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"ATTENZIONE: nel file {f} mancano colonne: {missing}")
            continue

        # Filtra errori
        dfe = df.copy()
        # Tratta errori come numerici, valori non numerici -> NaN -> riempiti con 0 per essere conservativi?
        # Qui consideriamo non validi solo valori numerici != 0.
        for ec in err_cols:
            # prova conversione a numerico
            dfe[ec] = pd.to_numeric(dfe[ec], errors="coerce").fillna(0.0)

        mask_ok = (dfe[err_cols[0]] == 0) & (dfe[err_cols[1]] == 0)
        dfe = dfe[mask_ok]

        # Rimuovi righe con NaN nelle colonne chiave
        dfe = dfe.replace([np.inf, -np.inf], np.nan)
        dfe = dfe.dropna(subset=[x_col, y_col, env_col, settings_col])

        # Converti x e y a float
        dfe[x_col] = pd.to_numeric(dfe[x_col], errors="coerce")
        dfe[y_col] = pd.to_numeric(dfe[y_col], errors="coerce")
        dfe = dfe.dropna(subset=[x_col, y_col])

        dfe[file_origin_col] = os.path.basename(f)
        frames.append(dfe)

    if not frames:
        print("Nessun dato valido trovato nei CSV forniti.")
        return

    data_all = pd.concat(frames, ignore_index=True)

    # Mappatura simboli per env_name
    unique_envs = sorted(data_all[env_col].astype(str).unique())
    markers_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>', 'h', 'H', 'd', 'p']
    env_to_marker: Dict[str, str] = {}
    for i, env in enumerate(unique_envs):
        env_to_marker[env] = markers_cycle[i % len(markers_cycle)]

    # Mappatura colori per settings
    unique_settings = sorted(data_all[settings_col].astype(str).unique())
    cmap = plt.get_cmap("tab20")
    settings_to_color: Dict[str, tuple] = {}
    for i, s in enumerate(unique_settings):
        settings_to_color[s] = cmap(i % cmap.N)

    # ---------- Plot complessivo ----------
    fig, ax = plt.subplots(figsize=figsize)
    # Scatter per combinazioni settings-env
    for s in unique_settings:
        df_s = data_all[data_all[settings_col].astype(str) == s]
        for e in unique_envs:
            df_se = df_s[df_s[env_col].astype(str) == e]
            if df_se.empty:
                continue
            ax.scatter(
                df_se[x_col], df_se[y_col],
                label=None,  # gestiamo legende custom
                s=48,
                marker=env_to_marker[e],
                color=settings_to_color[s],
                edgecolor="k",
                linewidths=0.5,
                alpha=0.9
            )

    # Fronte di Pareto globale
    pf = pareto_front(data_all, x_col=x_col, y_col=y_col)
    if not pf.empty:
        # ordina per x decrescente per tracciare una spezzata coerente
        pf = pf.sort_values(by=[x_col], ascending=False)
        ax.plot(pf[x_col], pf[y_col], color="black", linestyle="-", linewidth=2, label="Fronte Pareto")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)

    # Legenda simboli -> env_name
    env_legend_elems = [
        Line2D([0], [0], marker=env_to_marker[e], color='w', markerfacecolor='gray',
               markeredgecolor='k', linewidth=0, markersize=8, label=str(e))
        for e in unique_envs
    ]
    legend1 = ax.legend(handles=env_legend_elems, title=f"Micro ({env_col})", loc="upper right", frameon=True)
    ax.add_artist(legend1)

    # Legenda colori -> settings
    settings_legend_elems = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=settings_to_color[s],
               markeredgecolor='k', linewidth=0, markersize=8, label=str(s))
        for s in unique_settings
    ]
    legend2 = ax.legend(handles=settings_legend_elems, title=f"Settings ({settings_col})", loc="lower left", frameon=True)

    ax.grid(True, linestyle="--", alpha=0.3)
    # Assicura che la latenza non venga capovolta (minimo in basso)
    # Nota: non invertiamo gli assi; ottimizziamo visualmente con lim padding
    def pad_limits(vals, pad_ratio=0.05, lower_is_min=True):
        if len(vals) == 0:
            return None
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
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
    plt.close(fig)
    print(f"Salvato grafico complessivo in: {output_path}")

    # ---------- Plot per-file (opzionale, utile per confronto diretto) ----------
    out_dir = os.path.dirname(output_path) or "."
    base = os.path.splitext(os.path.basename(output_path))[0]
    for f in sorted(data_all[file_origin_col].unique()):
        dff = data_all[data_all[file_origin_col] == f]
        if dff.empty:
            continue
        figf, axf = plt.subplots(figsize=figsize)
        # Scatter per questo file
        svals = sorted(dff[settings_col].astype(str).unique())
        evals = sorted(dff[env_col].astype(str).unique())
        for s in svals:
            df_s = dff[dff[settings_col].astype(str) == s]
            for e in evals:
                df_se = df_s[df_s[env_col].astype(str) == e]
                if df_se.empty:
                    continue
                axf.scatter(
                    df_se[x_col], df_se[y_col],
                    s=48,
                    marker=env_to_marker.get(e, 'o'),
                    color=settings_to_color.get(s, 'C0'),
                    edgecolor="k",
                    linewidths=0.5,
                    alpha=0.9
                )
        # Fronte di Pareto per questo file
        pf_f = pareto_front(dff, x_col=x_col, y_col=y_col)
        if not pf_f.empty:
            pf_f = pf_f.sort_values(by=[x_col], ascending=False)
            axf.plot(pf_f[x_col], pf_f[y_col], color="black", linestyle="-", linewidth=2, label="Fronte Pareto")

        axf.set_xlabel(x_col)
        axf.set_ylabel(y_col)
        axf.set_title(f"{title} — {f}")

        # Legende
        env_legend_elems_f = [
            Line2D([0], [0], marker=env_to_marker.get(e, 'o'), color='w', markerfacecolor='gray',
                   markeredgecolor='k', linewidth=0, markersize=8, label=str(e))
            for e in evals
        ]
        legend1f = axf.legend(handles=env_legend_elems_f, title=f"Micro ({env_col})", loc="upper right", frameon=True)
        axf.add_artist(legend1f)
        settings_legend_elems_f = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=settings_to_color.get(s, 'C0'),
                   markeredgecolor='k', linewidth=0, markersize=8, label=str(s))
            for s in svals
        ]
        legend2f = axf.legend(handles=settings_legend_elems_f, title=f"Settings ({settings_col})", loc="lower left", frameon=True)

        axf.grid(True, linestyle="--", alpha=0.3)
        xlimf = pad_limits(dff[x_col].values)
        ylimf = pad_limits(dff[y_col].values)
        if xlimf: axf.set_xlim(xlimf)
        if ylimf: axf.set_ylim(ylimf)

        figf.tight_layout()
        out_file = os.path.join(out_dir, f"{base}__{os.path.splitext(f)[0]}.png")
        figf.savefig(out_file, dpi=dpi)
        plt.close(figf)
        print(f"Salvato grafico per-file in: {out_file}")

# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Plot Pareto Rate vs Latenza da CSV multipli.")
    p.add_argument("csv", nargs='+', help="Lista di file CSV di input.")
    p.add_argument("--out", default="pareto_overall.png", help="Percorso file PNG di output complessivo.")
    p.add_argument("--title", default="Pareto: Rate vs Latenza", help="Titolo del grafico.")
    p.add_argument("--xcol", default="Rate_DL_py_load_test_tflite_mcu", help="Nome colonna asse X (rate).")
    p.add_argument("--ycol", default="mean_tot_latency", help="Nome colonna asse Y (latenza).")
    p.add_argument("--envcol", default="env_name", help="Colonna micro/env per simboli.")
    p.add_argument("--settingscol", default="end_folder_Training_Size_dd_epochs", help="Colonna settings per colori.")
    p.add_argument("--errcols", nargs=2, default=["Error_does_not_fit", "Error_model_in_ram"], help="Due colonne di errore (saltare se != 0).")
    p.add_argument("--dpi", type=int, default=150, help="DPI del PNG.")
    p.add_argument("--figw", type=int, default=10, help="Larghezza figura in pollici.")
    p.add_argument("--figh", type=int, default=7, help="Altezza figura in pollici.")
    return p.parse_args()

def main():
    #args = parse_args()
    
    files = ['ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_nucleo-h753zi.csv', 'ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_esp32-s2-saola-tflm.csv']

    output_png = 'pareto_plot.png'
    
    #plot_pareto(
    #    csv_files=args.csv,
    #    output_path=args.out,
    #    x_col=args.xcol,
    #    y_col=args.ycol,
    #    env_col=args.envcol,
    #    settings_col=args.settingscol,
    #    err_cols=tuple(args.errcols),
    #    title=args.title,
    #    figsize=(args.figw, args.figh),
    #    dpi=args.dpi
    #)

    plot_pareto(
        csv_files=files,
        output_path=output_png,
        x_col="Rate_DL_py_load_test_tflite_mcu",
        y_col="mean_tot_latency",
        env_col="env_name",
        settings_col="end_folder_Training_Size_dd_epochs",
        err_cols=tuple(["Error_does_not_fit", "Error_model_in_ram"]),
        title="Pareto: Rate vs Latenza",
        figsize=(10,7),
        dpi=300
    )

if __name__ == "__main__":
    main()