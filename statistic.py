import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# 1. Naložimo shranjene podatke iz model_runs.json
history_file = "model_runs.json"

try:
    with open(history_file, "r") as f:
        runs_history = json.load(f)
except FileNotFoundError:
    print(f"Napaka: Datoteka '{history_file}' ne obstaja. Najprej zaženite nekaj modelov.")
    runs_history = []

if not runs_history:
    print("Ni zabeleženih zagonov v model_runs.json.")
else:
    df = pd.DataFrame(runs_history)
    st_modelov = len(df)
    df['accuracy'] = df['final_accuracy']
    
    print(f"Naloženih {st_modelov} zagonov modelov.")

    # ==========================================
    # ANALIZA IN VIZUALIZACIJA
    # ==========================================

    # 1. Natančnost vs. vsak dejavnik (3 paneli vzporedno)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    factors = [
        ("data_rows", "Število vrstic podatkov (Log skala)", True),
        ("epoch_count", "Število epoh (Log skala)", True),
        ("training_percent", "Odstotek za treniranje (%)", False)
    ]

    for ax, (col, title, use_log) in zip(axes, factors):
        sns.regplot(
            data=df, x=col, y="accuracy", ax=ax,
            scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'},
            lowess=True
        )
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("Natančnost (%)")
        if use_log:
            ax.set_xscale('log')

    plt.tight_layout()
    plt.show()

    # 2. Velikost vpliva (Standardized Regression Coefficients)
    predictors = ['data_rows', 'epoch_count', 'training_percent', 'eval_percent']
    df_std = df[['accuracy'] + predictors].apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x)

    X = sm.add_constant(df_std[predictors])
    y = df_std['accuracy']
    model = sm.OLS(y, X).fit()

    coefs = model.params.drop('const')
    conf_int = model.conf_int().drop('const')
    errors = [coefs - conf_int[0], conf_int[1] - coefs]

    slovenian_labels = {
        'data_rows': 'Število vrstic',
        'epoch_count': 'Število epoh',
        'training_percent': 'Odstotek treniranja',
        'eval_percent': 'Odstotek validacije'
    }
    coefs.index = coefs.index.map(slovenian_labels)

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(coefs))
    ax.barh(y_pos, coefs, xerr=errors, align='center', alpha=0.8, ecolor='black', capsize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coefs.index)
    ax.invert_yaxis()  
    ax.set_xlabel("Standardiziran regresijski koeficient")
    ax.set_title(f"Pomembnost dejavnikov (R² = {model.rsquared:.2f}, n = {st_modelov})")
    ax.axvline(0, color='grey', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.show()

    # 3. Toplotna mapa (Rows × Epochs, če je vsaj 50 zagonov)
    if st_modelov >= 50:
        plt.figure(figsize=(8, 6))
        df['rows_bin'] = pd.qcut(df['data_rows'], q=min(5, len(df)), labels=False, duplicates='drop')
        df['epoch_bin'] = pd.qcut(df['epoch_count'], q=min(5, len(df)), labels=False, duplicates='drop')
        
        pivot_table = df.pivot_table(values='accuracy', index='rows_bin', columns='epoch_bin', aggfunc='mean')
        
        # Popravljeno iz applymap na map
        annot_matrix = pivot_table.map(lambda val: f"{val:.2f}%" if pd.notnull(val) else "")
        
        sns.heatmap(
            pivot_table, annot=annot_matrix, fmt="", cmap='viridis', 
            cbar_kws={'label': 'Povprečna natančnost (%)'}
        )
        plt.title("Interakcija: Število vrstic × Število epoh (Povprečna natančnost)")
        plt.xlabel("Skupina epoh")
        plt.ylabel("Skupina vrstic podatkov")
        plt.show()
    else:
        print(f"\n[Opomba] Toplotna mapa je preskočena, ker imate {st_modelov} zagonov (potrebnih je vsaj 50).")

    # ==========================================
    # RAZLAGA REZULTATOV (V KONSOLI)
    # ==========================================
    print("\n" + "="*50)
    print("RAZLAGA REZULTATOV ANALIZE:")
    print("="*50)
    print("1. Natančnost vs. posamezni dejavnik (Graf 1):")
    print("   Rdeča ukrivljena črta (LOWESS) jasno prikazuje padajoče donose (diminishing returns).")
    print("\n2. Velikost vpliva in rangiranje (Graf 2):")
    print("   Prikazuje, kateri dejavnik najbolj vpliva na natančnost.")
    print(f"   Model pojasni {model.rsquared * 100:.1f}% variacije natančnosti (R² = {model.rsquared:.2f}).")
    if st_modelov >= 50:
        print("\n3. Toplotna mapa interakcij (Graf 3):")
        print("   Barva celic in izpisane vrednosti z znakom '%' predstavljajo točno povprečno natančnost.")
    print("="*50)