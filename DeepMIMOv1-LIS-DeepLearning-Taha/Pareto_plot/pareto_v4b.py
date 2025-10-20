#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script per la generazione di uno scatter plot Pareto
da più file CSV, confrontando rate e latenza di diversi microcontrollori.
L'ottimo è in basso a sinistra (latenza minima, rate massimo).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def compute_pareto_front(x, y):
    """
    Restituisce gli indici dei punti appartenenti al fronte di Pareto.
    Ottimo: latenza minima (y ↓), rate massimo (x ↑)
    """
    data = np.array(list(zip(x, y)))
    pareto_mask = np.ones(data.shape[0], dtype=bool)
    for i, point in enumerate(data):
        if pareto_mask[i]:
            # Un punto è dominato se esiste un altro con x>= e y<= (migliore in almeno una dimensione)
            dominates = np.any(
                (data[pareto_mask][:, 0] >= point[0]) & (data[pareto_mask][:, 1] <= point[1]) &
                ((data[pareto_mask][:, 0] > point[0]) | (data[pareto_mask][:, 1] < point[1])),
                axis=0
            )
            if dominates:
                pareto_mask[i] = False
    return np.where(pareto_mask)[0]


def plot_pareto_from_csvs(csv_files, output_file="pareto_plot.png"):
    """Genera un unico scatter plot Pareto da una lista di file CSV."""
    plt.figure(figsize=(10, 7))

    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    marker_map = {}
    color_map = {}

    all_data = []  # raccogli tutti i punti per pareto globale

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        # Filtra righe con errori
        df = df[(df["Error_does_not_fit"] == 0) & (df["Error_model_in_ram"] == 0)]

        if df.empty:
            print(f"[ATTENZIONE] Nessun dato valido in {csv_path}")
            continue

        # Colore per la configurazione
        setting = df["end_folder_Training_Size_dd_epochs"].iloc[0]
        if setting not in color_map:
            color_map[setting] = f"C{len(color_map)}"

        # Marker per tipo di micro (stesso per _modelinram e non)
        for env in df["env_name"].unique():
            base_micro = env.replace("_modelinram", "")
            if base_micro not in marker_map:
                marker_map[base_micro] = markers[len(marker_map) % len(markers)]

        # Plot punti
        for env in df["env_name"].unique():
            sub = df[df["env_name"] == env]
            x = sub["Rate_DL_py_load_test_tflite"]
            y = sub["mean_tot_latency"]
            base_micro = env.replace("_modelinram", "")
            marker = marker_map[base_micro]
            color = color_map[setting]

            if "modelinram" in env.lower():
                plt.scatter(x, y, marker=marker, facecolor=color, edgecolor='black',
                            s=70, alpha=0.9, label=f"{env} ({setting})")
            else:
                plt.scatter(x, y, marker=marker, facecolors='none', edgecolors=color,
                            s=70, alpha=0.9, label=f"{env} ({setting})")

            all_data.append(sub)

    # --- Fronte di Pareto globale (su tutti i file) ---
    if all_data:
        df_all = pd.concat(all_data, ignore_index=True)
        x_all = df_all["Rate_DL_py_load_test_tflite"].values
        y_all = df_all["mean_tot_latency"].values
        pareto_idx = compute_pareto_front(x_all, y_all)

        pareto_points = df_all.iloc[pareto_idx].sort_values(
            by="Rate_DL_py_load_test_tflite", ascending=True
        )
        plt.plot(pareto_points["Rate_DL_py_load_test_tflite"],
                 pareto_points["mean_tot_latency"],
                 'k-', linewidth=2, label="Pareto front")

    # --- Legende ---
    symbol_legend = [
        plt.Line2D([0], [0], marker=marker_map[m], color='k', linestyle='None', label=m)
        for m in marker_map.keys()
    ]
    color_legend = [
        plt.Line2D([0], [0], color=color_map[c], marker='o', linestyle='None', label=c)
        for c in color_map.keys()
    ]
    pareto_legend = [
        plt.Line2D([0], [0], color='k', linewidth=2, label='Pareto front')
    ]

    plt.xlabel("Rate (Rate_DL_py_load_test_tflite)")
    plt.ylabel("Latenza (mean_tot_latency)")
    plt.title("Pareto plot - rate vs latenza")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(
        handles=symbol_legend + color_legend + pareto_legend,
        loc='best', fontsize=8, title="Legenda"
    )

    # Inverti asse X (rate alti a sinistra)
    plt.gca().invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"[INFO] Plot salvato in: {output_file}")


def main():
    csv_files = [
        'ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_nucleo-h753zi.csv',
        'ok_profiling_seed0_grid1200_M3232_Mbar8_30000_ep200_esp32-s2-saola-tflm.csv'
    ]

    plot_pareto_from_csvs(csv_files, output_file="pareto_plot.png")


if __name__ == "__main__":
    main()
