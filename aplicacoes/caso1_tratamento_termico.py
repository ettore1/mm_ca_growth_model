"""
Caso 1 — Otimização de Tratamento Térmico
==========================================
Varre diferentes valores de P_NEUTRAL (proxy fenomenológico de mobilidade de
contorno) e calcula as curvas de crescimento de grão resultantes. Permite
identificar as condições de temperatura × tempo necessárias para atingir
tamanho de grão alvo (normas ASTM), a partir de calibração com dados
experimentais.

Conceito de calibração física
─────────────────────────────
  k_CA [px²/CAS] × (px_to_um)² = k_phys [µm²/s] × CAS_to_s

  Dados de referência (exemplo — aço de baixo carbono a 700 °C):
    1 px = 5 µm  →  grade 200 × 200 px ≡ 1,0 mm × 1,0 mm
    k_CA ≈ 0,232 px²/CAS  (ajustado no script principal a T_NOISE = 0,5)
    k_phys ≈ 3,0 µm²/s    (Humphreys & Hatherly 2004, aço com soluto leve)
    → 1 CAS ≈ 1,93 s de recozimento real a 700 °C

P_NEUTRAL → temperatura (relação de Arrhenius)
───────────────────────────────────────────────
  P_NEUTRAL é tratado como proxy de mobilidade de contorno M (Seção 2.1 do
  relatório: v = M·P). A mobilidade de contorno segue lei de Arrhenius
  M(T) = M0 * exp(-Q/(R*T)), com Q ≈ 172,4 kJ/mol — energia de ativação
  para crescimento de grão em aço ferrítico (Oliveira, Sandim & Raabe 2017,
  J. Nucl. Mater. 485, 23-38). Invertendo a relação com ancoragem em
  P_NEUTRAL = 0,5 ↔ T_ref = 700 °C (973,15 K), obtém-se T(P_NEUTRAL) para
  cada ponto da varredura — uma faixa bem mais estreita (~96 °C) e
  fisicamente mais plausível que um mapeamento linear arbitrário.

Baseado em:
  simulacao_ca_classica.py — motor de simulação CA 2D com regra de mínima
  energia e aceitação estocástica (He et al. 2006, Raghavan & Sahay 2007).
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

# ── Localiza o motor CA no diretório pai ────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # aplicacoes/
PARENT    = os.path.dirname(_THIS_DIR)                   # simulacao_ca_classica/
sys.path.insert(0, PARENT)
import simulacao_ca_classica as ca

# ── Parâmetros de simulação ──────────────────────────────────────────────────
GRID_SIZE    = 200
N_NUCLEI     = 600
N_STEPS      = 1500    # passos CA por temperatura
RECORD_EVERY = 100
SEED         = 42

# Valores de P_NEUTRAL a varrer (proxy de mobilidade de contorno)
P_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]

# Mapeamento P_NEUTRAL → temperatura via relação de Arrhenius para
# mobilidade de contorno M(T) = M0*exp(-Q/(R*T)), com P_NEUTRAL como proxy
# de M. Q = energia de ativação para crescimento de grão em aço ferrítico
# (Oliveira, Sandim & Raabe 2017). Ancoragem: P_NEUTRAL=0,5 ↔ T_ref=700°C.
Q_ATIVACAO = 172400.0    # J/mol (Oliveira et al. 2017, Eurofer-97 ferrítico)
R_GAS      = 8.314       # J/(mol*K)
P_REF      = 0.5
T_REF_K    = 700.0 + 273.15   # 973,15 K

def t_phys(p_neutral):
    """Temperatura (°C) cuja mobilidade de Arrhenius reproduz P_NEUTRAL,
    ancorada em P_NEUTRAL=0,5 <-> T_ref=700°C."""
    inv_T = 1.0 / T_REF_K - (R_GAS / Q_ATIVACAO) * np.log(p_neutral / P_REF)
    return 1.0 / inv_T - 273.15   # K -> °C

# ── Calibração física ────────────────────────────────────────────────────────
PX_TO_UM   = 5.0    # 1 pixel = 5 µm
K_CA_REF   = 0.232  # px²/CAS   (T_NOISE = 0,5, ajustado no script principal)
K_PHYS_REF = 3.0    # µm²/s     (referência experimental a 700 °C)
CAS_TO_S   = K_CA_REF * PX_TO_UM**2 / K_PHYS_REF  # ≈ 1,93 s/CAS

# Tamanhos de grão alvo (diâmetro em µm) — norma ASTM aproximada
TARGET_D_UM  = [30,  50,  80]
TARGET_LABEL = ['ASTM 7 (~30 µm)', 'ASTM 5 (~50 µm)', 'ASTM 3 (~80 µm)']

OUTPUT_DIR = os.path.join(_THIS_DIR, 'resultados')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Funções auxiliares ───────────────────────────────────────────────────────

def run_single(p_neutral: float):
    """Executa uma simulação completa com P_NEUTRAL dado.

    Retorna array (n_pontos, 3): [step, R_mean_px, A_mean_px²]
    """
    ca.T_NOISE   = 0.5
    ca.P_NEUTRAL = p_neutral
    np.random.seed(SEED)

    grid = ca.initialize_grid(GRID_SIZE, N_NUCLEI, SEED)
    s0   = ca.grain_statistics(grid)
    hist = [(0, s0['mean_radius'], s0['mean_area'])]

    for step in range(1, N_STEPS + 1):
        grid = ca.ca_step(grid)
        if step % RECORD_EVERY == 0:
            s = ca.grain_statistics(grid)
            hist.append((step, s['mean_radius'], s['mean_area']))

    return np.array(hist)   # (n, 3)


def fit_burke(steps, areas):
    """Ajusta Lei de Burke ΔA = k·t; retorna (k, R²)."""
    dA   = areas - areas[0]
    mask = (steps > 50) & (dA > 0)
    if mask.sum() < 4:
        return np.nan, np.nan
    popt, _ = curve_fit(lambda t, k: k * t, steps[mask], dA[mask], p0=[0.1])
    k = float(popt[0])
    y_fit  = k * steps[mask]
    ss_res = np.sum((dA[mask] - y_fit) ** 2)
    ss_tot = np.sum((dA[mask] - dA[mask].mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return k, r2


def fit_growth_exp(steps, radii):
    """Ajusta lei de potência R ~ t^n em escala log-log; retorna (n, R²)."""
    mask = steps > 80
    if mask.sum() < 4:
        return np.nan, np.nan
    slope, intercept, rv, _, _ = stats.linregress(
        np.log(steps[mask]), np.log(radii[mask]))
    return float(slope), float(rv ** 2)


# ── Programa principal ───────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  CASO 1 — Otimização de Tratamento Térmico")
    print(f"  Grade: {GRID_SIZE}×{GRID_SIZE}  |  {N_NUCLEI} núcleos  |  {N_STEPS} passos/T")
    print(f"  Calibração: 1 px = {PX_TO_UM} µm  |  1 CAS ≈ {CAS_TO_S:.2f} s a 700 °C")
    print("=" * 64)

    results  = {}
    k_burke  = {}

    for P in P_VALUES:
        print(f"\n  Simulando P_NEUTRAL = {P:.1f}  (~{t_phys(P):.0f} °C) ...", flush=True)
        hist = run_single(P)
        results[P] = hist
        k, r2 = fit_burke(hist[:, 0], hist[:, 2])
        n, r2n = fit_growth_exp(hist[:, 0], hist[:, 1])
        k_burke[P] = k
        d_f = 2.0 * hist[-1, 1] * PX_TO_UM
        print(f"    k_CA = {k:.4f} px²/CAS  |  n = {n:.3f}  |  d_final = {d_f:.1f} µm")

    # ── Figura: 2 painéis ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(P_VALUES)))

    # Painel esquerdo — Curvas de crescimento (d vs. tempo)
    ax = axes[0]
    for P, col in zip(P_VALUES, colors):
        hist  = results[P]
        t_s   = hist[:, 0] * CAS_TO_S         # CAS → segundos
        d_um  = 2.0 * hist[:, 1] * PX_TO_UM   # raio px → diâmetro µm
        label = f'P_NEUTRAL = {P}  (~{t_phys(P):.0f} °C)'
        ax.plot(t_s, d_um, '-o', markersize=4, color=col, lw=1.8, label=label)

    # Linhas-alvo ASTM (linhas tracejadas horizontais)
    for d_tgt, lbl in zip(TARGET_D_UM, TARGET_LABEL):
        ax.axhline(d_tgt, ls='--', color='dimgray', lw=1.0, alpha=0.8)
        ax.text(t_s[2], d_tgt + 1.5, lbl, fontsize=8.5, color='dimgray', va='bottom')

    ax.set_xlabel('Tempo de recozimento (s)', fontsize=11)
    ax.set_ylabel('Diâmetro médio de grão (µm)', fontsize=11)
    ax.set_title('Crescimento de Grãos × Tempo\n(linhas tracejadas = tamanho-alvo ASTM)',
                 fontsize=11)
    ax.legend(fontsize=8.5, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Eixo superior em CAS (passos de simulação)
    ax_top = ax.twiny()
    ax_top.set_xlim(0, N_STEPS * CAS_TO_S)
    cas_ticks = np.arange(0, N_STEPS + 1, 300)
    ax_top.set_xticks(cas_ticks * CAS_TO_S)
    ax_top.set_xticklabels([str(c) for c in cas_ticks], fontsize=8)
    ax_top.set_xlabel('Passos CA (CAS)', fontsize=9)

    # Painel direito — k × P_NEUTRAL (análoga à curva de Arrhenius)
    ax = axes[1]
    P_arr = np.array(P_VALUES)
    k_arr = np.array([k_burke[P] * PX_TO_UM**2 / CAS_TO_S for P in P_VALUES])
    ax.plot(P_arr, k_arr, '-', color='steelblue', lw=1.5, zorder=3)
    ax.scatter(P_arr, k_arr, s=100, zorder=5, color='steelblue',
               edgecolors='navy', linewidths=0.8)
    for P, k, col in zip(P_arr, k_arr, colors):
        ax.annotate(f'{t_phys(P):.0f} °C\nk={k:.1f}',
                    (P, k), textcoords='offset points',
                    xytext=(8, -4), fontsize=8.5, color='navy')

    ax.set_xlabel('P_NEUTRAL (mobilidade de contornos planos)', fontsize=11)
    ax.set_ylabel('Constante cinética k  (µm²/s)', fontsize=11)
    ax.set_title('Constante de Burke × Temperatura\n(quanto maior P_NEUTRAL, maior a taxa de crescimento)',
                 fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Caso 1 — Otimização de Tratamento Térmico\n'
        f'(escala: 1 px = {PX_TO_UM} µm  |  ref.: P_NEUTRAL 0,5 ↔ 700 °C, k = {K_PHYS_REF} µm²/s)',
        fontsize=11)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, 'caso1_tratamento_termico.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\n  Figura salva: {out}")

    # ── Tabela de resultados ─────────────────────────────────────────────────
    print(f"\n  {'P_NEUTRAL':>9} {'T(°C)':>7} {'k_CA(px²/CAS)':>15} "
          f"{'k(µm²/s)':>10} {'d_f(µm)':>10}")
    print("  " + "-" * 55)
    for P in P_VALUES:
        h   = results[P]
        k_c = k_burke[P]
        k_p = k_c * PX_TO_UM**2 / CAS_TO_S
        d_f = 2.0 * h[-1, 1] * PX_TO_UM
        print(f"  {P:>9.1f} {t_phys(P):>7.0f} {k_c:>15.4f} {k_p:>10.2f} {d_f:>10.1f}")

    print("\n  COMO USAR PARA SEU MATERIAL:")
    print("  1) Meça d₀ (µm) e o raio médio inicial na microscopia")
    print("     ajuste PX_TO_UM = d₀_exp / (2 × R₀_simulado)")
    print("  2) Faça um recozimento experimental e meça d(t) em dois instantes")
    print("     ajuste K_PHYS_REF = (d²(t₂) - d²(t₁)) / (2 × (t₂ - t₁))")
    print("  3) Recalcule CAS_TO_S = K_CA_REF × PX_TO_UM² / K_PHYS_REF")
    print("  4) Use o gráfico esquerdo para estimar tempo necessário ao d-alvo")


if __name__ == '__main__':
    main()
