#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulacao de Crescimento de Graos por Automatos Celulares 2D
=============================================================
Modelo CA classico com aceitacao estocastica leve.

PROBLEMA DA VERSAO ANTERIOR (simulacao_ca_graos_old.py):
  A regra deterministica pura (aceitar apenas ΔE < 0) congelava os
  contornos planos (ΔE = 0 → nunca aceitos), resultando em graos com
  faces retas e geometria quadrada/poligonal sem curvatura.

SOLUCAO IMPLEMENTADA:
  Mantem a regra CA (proposta = vizinho de MINIMA energia — deterministica),
  mas adiciona aceitacao probabilistica para:
    ΔE < 0  →  P = 1.0          (sempre aceito — reduz energia)
    ΔE = 0  →  P = P_NEUTRAL    (contornos planos podem migrar)
    ΔE > 0  →  P = exp(-ΔE/T)  (flutuacoes termicas leves)

  Isso e suficiente para produzir contornos curvos e crescimento normal,
  preservando o carater de Automato Celular: a orientacao proposta e uma
  funcao deterministica do estado da vizinhanca (vizinho otimo), distinguindo
  esta abordagem do modelo Monte Carlo Potts (vizinho aleatorio).

Baseado em:
  - He et al. (2006), Mater. Sci. Eng. A 429, 236-246
  - Raghavan & Sahay (2007), Mater. Sci. Eng. A 445-446, 203-209
  - Liu, Baudin & Penelle (1996), Scripta Materialia 34, 1679-1683

Vizinhanca de Moore (8 vizinhos), condicoes de contorno periodicas.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import weibull_min, lognorm
from scipy.optimize import curve_fit
from scipy import stats
import os
import time

# ─────────────────────── PARAMETROS ────────────────────────
GRID_SIZE    = 200     # Tamanho da grade N×N (celulas)
N_NUCLEI     = 600     # Numero de graos iniciais
N_STEPS      = 2500    # Numero de passos CA (CAS)
T_NOISE      = 0.5     # Temperatura de ruido termico (para ΔE > 0)
P_NEUTRAL    = 0.5     # Probabilidade de aceitar movimentos neutros (ΔE = 0)
RECORD_EVERY = 100     # Registrar estatisticas a cada N passos
SNAP_STEPS   = {0, 300, 750, 1250, 1750, 2500}  # Passos para captura de snapshot
SEED         = 42
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))
# ────────────────────────────────────────────────────────────

np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 120,
})


# ===========================================================
# 1. INICIALIZACAO — Diagrama de Voronoi
# ===========================================================

def initialize_grid(N: int, n_nuclei: int, seed: int = 42) -> np.ndarray:
    """
    Cria microestrutura inicial via diagrama de Voronoi.
    Cada celula recebe o ID do grao (nucleo) mais proximo.
    """
    rng = np.random.RandomState(seed)
    px = rng.randint(0, N, n_nuclei)
    py = rng.randint(0, N, n_nuclei)
    nuclei = np.column_stack([px, py])
    tree = cKDTree(nuclei)

    xx, yy = np.mgrid[0:N, 0:N]
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    _, idx = tree.query(pts)
    grid = (idx + 1).reshape(N, N).astype(np.int32)
    return grid


# ===========================================================
# 2. PASSO DO AUTOMATO CELULAR
# ===========================================================

def get_moore_neighbors(grid: np.ndarray) -> np.ndarray:
    """
    Retorna array (8, N, N) com os 8 vizinhos de Moore.
    Condicao de contorno periodica (wrap).
    """
    p = np.pad(grid, 1, mode='wrap')
    return np.stack([
        p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:],
        p[1:-1, :-2],               p[1:-1, 2:],
        p[2:, :-2],  p[2:, 1:-1],  p[2:, 2:]
    ])  # (8, N, N)


def ca_step(grid: np.ndarray) -> np.ndarray:
    """
    Um passo do automato celular com aceitacao estocastica.

    Regra CA (deterministica):
      A orientacao proposta para cada celula de contorno e a do vizinho que
      MINIMIZA a energia local (numero de vizinhos discordantes). Esta e a
      regra deterministica de minima energia de He et al. (2006), que distingue
      este modelo do MC Potts (que seleciona um vizinho aleatorio).

    Aceitacao estocastica leve:
      ΔE < 0  →  P = 1.0           — sempre aceito
      ΔE = 0  →  P = P_NEUTRAL     — permite migracao de contornos planos
      ΔE > 0  →  P = exp(-ΔE/T)   — flutuacoes termicas (ancoragem reduzida)

    O motivo fisico: contornos planos possuem ΔE = 0 para qualquer reorientacao
    local. Sem aceitacao de movimentos neutros, esses contornos ficam congelados
    e os graos desenvolvem faces retas. Com P_NEUTRAL > 0, os contornos planos
    desenvolvem rugosidade termica, que e entao eliminada pela forca de curvatura,
    resultando em crescimento normal com contornos curvos.
    """
    N = grid.shape[0]
    nbrs = get_moore_neighbors(grid)       # (8, N, N)

    # Celulas de contorno: ao menos um vizinho com orientacao diferente
    is_boundary = np.any(nbrs != grid[np.newaxis], axis=0)  # (N, N)

    # Energia atual: numero de vizinhos DIFERENTES
    current_diff = np.sum(nbrs != grid[np.newaxis], axis=0).astype(np.float32)

    # Energia hipotetica se adotarmos a orientacao do vizinho k
    cand_diff = np.stack([
        np.sum(nbrs != nbrs[k][np.newaxis], axis=0)
        for k in range(8)
    ]).astype(np.float32)  # (8, N, N)

    # Regra CA: orientacao do vizinho de minima energia (proposta deterministica)
    # Desempate ALEATORIO: entre todos os vizinhos com a mesma energia minima,
    # um e selecionado uniformemente ao acaso. Isso e essencial para evitar o
    # vies direcional (contornos na diagonal 45 graus) que ocorre quando
    # np.argmin escolhe sempre o indice 0 (vizinho superior-esquerdo) em empates.
    min_diff   = cand_diff.min(axis=0, keepdims=True)          # (1, N, N)
    tied_mask  = (cand_diff == min_diff).astype(np.float32)    # (8, N, N)
    rand_tie   = np.random.random((8, N, N)).astype(np.float32) * tied_mask
    best_k      = np.argmax(rand_tie, axis=0)                  # (N, N)
    ii          = np.arange(N)[:, np.newaxis]
    jj          = np.arange(N)[np.newaxis, :]
    best_orient = nbrs[best_k, ii, jj]                         # (N, N)
    best_diff   = cand_diff[best_k, ii, jj]                    # (N, N)

    delta_E = best_diff - current_diff  # (N, N)

    # Probabilidade de aceitacao estocastica
    prob = np.where(
        delta_E < 0,  1.0,
        np.where(delta_E == 0, P_NEUTRAL,
                 np.exp(-delta_E / T_NOISE))
    ).astype(np.float32)

    rand   = np.random.random((N, N)).astype(np.float32)
    accept = is_boundary & (best_orient != grid) & (rand < prob)


    new_grid = grid.copy()
    new_grid[accept] = best_orient[accept]
    return new_grid


# ===========================================================
# 3. ESTATISTICAS DE GRAOS
# ===========================================================

def grain_statistics(grid: np.ndarray) -> dict:
    """Calcula estatisticas microestruturais do estado atual."""
    _, counts = np.unique(grid, return_counts=True)
    radii = np.sqrt(counts / np.pi)
    return {
        'n_grains'   : int(len(counts)),
        'mean_area'  : float(np.mean(counts)),
        'mean_radius': float(np.mean(radii)),
        'radii'      : radii,
    }


# ===========================================================
# 4. LOOP PRINCIPAL DE SIMULACAO
# ===========================================================

def run_simulation():
    print("=" * 64)
    print("  SIMULACAO CA CLASSICO — CRESCIMENTO DE GRAOS 2D")
    print(f"  Grade: {GRID_SIZE}x{GRID_SIZE} | Graos iniciais: {N_NUCLEI}")
    print(f"  Passos: {N_STEPS} | T_noise={T_NOISE} | P_neutral={P_NEUTRAL}")
    print("=" * 64)

    grid = initialize_grid(GRID_SIZE, N_NUCLEI, SEED)

    history   = []
    snapshots = {}

    s = grain_statistics(grid)
    s['step'] = 0
    history.append(s)
    snapshots[0] = grid.copy()
    print(f"  CAS    0 | Graos: {s['n_grains']:4d} | R_med = {s['mean_radius']:.2f}")

    t0 = time.time()
    for step in range(1, N_STEPS + 1):
        grid = ca_step(grid)

        if step % RECORD_EVERY == 0 or step in SNAP_STEPS:
            s = grain_statistics(grid)
            s['step'] = step
            if step % RECORD_EVERY == 0:
                history.append(s)
                elapsed = time.time() - t0
                print(f"  CAS {step:4d} | Graos: {s['n_grains']:4d} | "
                      f"R_med = {s['mean_radius']:6.2f} | t={elapsed:.1f}s")

        if step in SNAP_STEPS:
            snapshots[step] = grid.copy()

    total = time.time() - t0
    print(f"\n  Simulacao concluida em {total:.1f}s")
    print("=" * 64)

    snap_stats = {st: grain_statistics(snapshots[st])
                  for st in sorted(snapshots.keys()) if st > 0}
    # Para analise de distribuicao, usa o snap com mais graos onde R_med
    # ja cresceu pelo menos 50% em relacao ao inicio (regime de crescimento)
    R_initial = history[0]['mean_radius']
    grown_steps = [st for st, ss in snap_stats.items()
                   if ss['mean_radius'] >= 1.5 * R_initial]
    if grown_steps:
        best_distrib_step = max(grown_steps, key=lambda s: snap_stats[s]['n_grains'])
    else:
        best_distrib_step = sorted(snap_stats.keys())[-1]
    distrib_radii = snap_stats[best_distrib_step]['radii']
    print(f"  Passo para distribuicao: {best_distrib_step} "
          f"({snap_stats[best_distrib_step]['n_grains']} graos)")
    return history, snapshots, distrib_radii, snap_stats, best_distrib_step


# ===========================================================
# 5. FIGURAS
# ===========================================================

def _save(fig, fname):
    path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Salvo: {fname}")
    return path


def _make_rgb_image(grid):
    """Converte grade de IDs de grao em imagem RGB colorida."""
    rng = np.random.RandomState(7)
    ids = np.unique(grid)
    color_map = {gid: rng.rand(3) for gid in ids}
    img = np.zeros((*grid.shape, 3))
    for gid, color in color_map.items():
        img[grid == gid] = color
    return img


def fig1_microestrutura(snapshots):
    """Evolucao colorida da microestrutura em 6 instantes."""
    steps = sorted(snapshots.keys())[:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()

    for idx, step in enumerate(steps):
        s = grain_statistics(snapshots[step])
        img = _make_rgb_image(snapshots[step])
        axes[idx].imshow(img, interpolation='nearest')
        axes[idx].set_title(
            f't = {step} CAS  |  {s["n_grains"]} graos  |  R_med={s["mean_radius"]:.1f}px',
            fontsize=10)
        axes[idx].axis('off')

    for idx in range(len(steps), len(axes)):
        axes[idx].axis('off')

    fig.suptitle(
        'Evolucao da Microestrutura — CA Classico com Aceitacao Estocastica\n'
        f'(T={T_NOISE}, P_neutral={P_NEUTRAL})',
        fontsize=12, fontweight='bold')
    fig.tight_layout()
    return _save(fig, 'fig1_microestrutura_evolucao.png')


def fig2_contornos(snapshots):
    """Contornos de grao em preto sobre fundo branco."""
    steps_show = [s for s in [300, 1250, 2500] if s in snapshots]
    if not steps_show:
        steps_show = sorted(snapshots.keys())[1:4]

    fig, axes = plt.subplots(1, len(steps_show), figsize=(14, 5))
    if len(steps_show) == 1:
        axes = [axes]

    for ax, step in zip(axes, steps_show):
        grid = snapshots[step]
        nbrs = get_moore_neighbors(grid)
        boundary = np.any(nbrs != grid[np.newaxis], axis=0)
        img = np.ones((*grid.shape, 3))
        img[boundary] = [0, 0, 0]
        ax.imshow(img, interpolation='nearest')
        ax.set_title(f't = {step} CAS', fontsize=12, fontweight='bold')
        ax.axis('off')

    fig.suptitle('Evolucao dos Contornos de Grao — CA Classico',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    return _save(fig, 'fig2_contornos_graos.png')


def fig3_cinetica(history):
    """Cinetica de crescimento: Lei de Burke + expoente de crescimento."""
    steps  = np.array([h['step'] for h in history], dtype=float)
    A_mean = np.array([h['mean_area']   for h in history])
    R_mean = np.array([h['mean_radius'] for h in history])
    A0 = A_mean[0]
    fit_params = {}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Painel 1: Lei de Burke (ΔA = k*t)
    dA   = A_mean - A0
    mask = steps > 50

    def burke(t, k):
        return k * t

    try:
        popt, _ = curve_fit(burke, steps[mask], dA[mask])
        k_b = float(popt[0])
        t_f = np.linspace(steps[mask][0], steps[-1], 300)
        ss_res = np.sum((dA[mask] - burke(steps[mask], k_b))**2)
        ss_tot = np.sum((dA[mask] - dA[mask].mean())**2)
        r2_b = 1 - ss_res / ss_tot
        fit_params.update({'k_burke': k_b, 'r2_burke': r2_b})
        ax1.plot(t_f, burke(t_f, k_b), 'r-', lw=2.5,
                 label=f'Burke: k={k_b:.4f}\n$R^2={r2_b:.4f}$')
    except Exception as e:
        print(f"  Burke fit falhou: {e}")

    ax1.scatter(steps, dA, color='steelblue', s=40, zorder=5, label='Simulacao CA')
    ax1.set_xlabel('Tempo (CAS)')
    ax1.set_ylabel(r'$\bar{A} - A_0$ (cel$^2$)')
    ax1.set_title(r'Lei de Burke: $\bar{A} - A_0 = k \cdot t$')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0); ax1.set_ylim(bottom=0)

    # Painel 2: Expoente de crescimento log-log
    mask2 = steps > 80
    try:
        slope, intercept, rv, _, _ = stats.linregress(
            np.log(steps[mask2]), np.log(R_mean[mask2]))
        m_est = 1.0 / slope if slope != 0 else None
        t_f2  = np.linspace(steps[mask2][0], steps[-1], 300)
        fit_params.update({'growth_exp': slope, 'r2_exp': rv**2,
                           'm_estimated': m_est})
        ax2.loglog(t_f2, np.exp(intercept) * t_f2**slope, 'r-', lw=2.5,
                   label=fr'$n={slope:.3f}$, $m_{{est}}={m_est:.2f}$'
                         fr', $R^2={rv**2:.4f}$')
    except Exception as e:
        print(f"  Expoente fit falhou: {e}")

    ax2.loglog(steps[steps > 0], R_mean[steps > 0],
               'o', color='darkorange', ms=5, zorder=5, label='Simulacao CA')
    ax2.set_xlabel('Tempo (CAS)')
    ax2.set_ylabel(r'$\bar{R}$ (cel)')
    ax2.set_title(r'Expoente de Crescimento ($\bar{R} \propto t^n$, $n_{teo}=0{,}5$)')
    ax2.legend(); ax2.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    _save(fig, 'fig3_cinetica_crescimento.png')
    return fit_params


def fig4_distribuicao(snap_stats, distrib_radii, distrib_step):
    """Distribuicao normalizada de tamanho de graos + auto-similaridade."""
    R_norm = distrib_radii / distrib_radii.mean()
    fit_params = {}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    bins  = np.linspace(0, 4.0, 28)
    x_fit = np.linspace(0.01, 4.0, 300)

    h, edges = np.histogram(R_norm, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_g = snap_stats[distrib_step]['n_grains']
    ax1.bar(centers, h, width=edges[1] - edges[0],
            alpha=0.55, color='steelblue', edgecolor='white',
            label=f'CA (passo {distrib_step}, {n_g} graos)')

    try:
        wc, wloc, wscale = weibull_min.fit(R_norm, floc=0)
        fit_params.update({'weibull_k': wc, 'weibull_scale': wscale})
        ax1.plot(x_fit, weibull_min.pdf(x_fit, wc, wloc, wscale), 'r-', lw=2.5,
                 label=f'Weibull (k={wc:.2f}, $\\lambda$={wscale:.2f})')
    except Exception as e:
        print(f"  Weibull fit falhou: {e}")

    try:
        ls, lloc, lscale = lognorm.fit(R_norm, floc=0)
        fit_params.update({'lognorm_sigma': ls})
        ax1.plot(x_fit, lognorm.pdf(x_fit, ls, lloc, lscale), 'g--', lw=2.5,
                 label=f'Log-normal ($\\sigma$={ls:.2f})')
    except Exception as e:
        print(f"  Lognorm fit falhou: {e}")

    ax1.set_xlabel(r'$R / \bar{R}$')
    ax1.set_ylabel('Densidade de probabilidade')
    ax1.set_title(f'Distribuicao Normalizada — passo {distrib_step}')
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_xlim(0, 4)

    steps_show = sorted(snap_stats.keys())[-4:]
    palette = ['royalblue', 'darkorange', 'green', 'crimson']
    for color, step in zip(palette, steps_show):
        r   = snap_stats[step]['radii']
        r_n = r / r.mean()
        ax2.hist(r_n, bins=bins, density=True, alpha=0.45,
                 color=color, edgecolor='none', label=f't = {step} CAS')

    ax2.set_xlabel(r'$R / \bar{R}$')
    ax2.set_ylabel('Densidade de probabilidade')
    ax2.set_title('Auto-Similaridade Temporal da Distribuicao')
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_xlim(0, 4)

    fig.tight_layout()
    _save(fig, 'fig4_distribuicao_tamanho.png')
    return fit_params


def fig5_evolucao_graos(history):
    """Numero de graos e raio medio ao longo do tempo."""
    steps  = np.array([h['step'] for h in history])
    n_gr   = np.array([h['n_grains']    for h in history])
    R_mean = np.array([h['mean_radius'] for h in history])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(steps, n_gr, 'o-', color='darkred', lw=2, ms=5)
    ax1.set_xlabel('Tempo (CAS)'); ax1.set_ylabel('Numero de Graos')
    ax1.set_title('Reducao do Numero de Graos com o Tempo')
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, R_mean, 's-', color='teal', lw=2, ms=5)
    ax2.set_xlabel('Tempo (CAS)'); ax2.set_ylabel(r'$\bar{R}$ (cel)')
    ax2.set_title('Crescimento do Raio Medio com o Tempo')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save(fig, 'fig5_evolucao_graos.png')


def fig6_cdf_validacao(snap_stats):
    """CDF normalizada em multiplos instantes para validacao de auto-similaridade."""
    steps_show = sorted(snap_stats.keys())
    palette    = ['royalblue', 'darkorange', 'green', 'crimson', 'purple', 'brown']

    fig, ax = plt.subplots(figsize=(9, 6))
    for color, step in zip(palette, steps_show):
        r   = snap_stats[step]['radii']
        r_n = np.sort(r / r.mean())
        cdf = np.arange(1, len(r_n) + 1) / len(r_n)
        ax.plot(r_n, cdf, lw=2, color=color, label=f't = {step} CAS')

    ax.set_xlabel(r'$R / \bar{R}$'); ax.set_ylabel('Fracao cumulativa F')
    ax.set_title('CDF Normalizada — Auto-Similaridade Temporal\n'
                 '(superposicao das curvas confirma regime estacionario)')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3.5); ax.set_ylim(0, 1)

    fig.tight_layout()
    return _save(fig, 'fig6_cdf_validacao.png')


# ===========================================================
# 6. MAIN
# ===========================================================

if __name__ == '__main__':
    history, snapshots, distrib_radii, snap_stats, distrib_step = run_simulation()

    print("\nGerando figuras...")
    fig1_microestrutura(snapshots)
    fig2_contornos(snapshots)
    fk = fig3_cinetica(history)
    fd = fig4_distribuicao(snap_stats, distrib_radii, distrib_step)
    fig5_evolucao_graos(history)
    fig6_cdf_validacao(snap_stats)

    print("\n" + "=" * 64)
    print("  RESULTADOS — CA CLASSICO")
    print("=" * 64)
    print(f"  Grade: {GRID_SIZE}x{GRID_SIZE}  |  Graos iniciais: {N_NUCLEI}")
    print(f"  T_noise={T_NOISE}  |  P_neutral={P_NEUTRAL}  |  Passos: {N_STEPS}")
    print("-" * 64)
    print(f"  Burke k          : {fk.get('k_burke', 'N/A'):.4f} cel^2/CAS")
    print(f"  Burke R2         : {fk.get('r2_burke', 'N/A'):.4f}")
    print(f"  Expoente n       : {fk.get('growth_exp', 'N/A'):.4f}")
    print(f"  m estimado       : {fk.get('m_estimated', 'N/A'):.2f}")
    print(f"  R2 log-log       : {fk.get('r2_exp', 'N/A'):.4f}")
    print(f"  Weibull k        : {fd.get('weibull_k', 'N/A'):.3f}")
    print(f"  Weibull lambda   : {fd.get('weibull_scale', 'N/A'):.3f}")
    print(f"  Log-normal sigma : {fd.get('lognorm_sigma', 'N/A'):.3f}")
    print("-" * 64)
    print(f"  Graos final      : {history[-1]['n_grains']}")
    print(f"  R_med inicial    : {history[0]['mean_radius']:.3f}px")
    print(f"  R_med final      : {history[-1]['mean_radius']:.3f}px")
    print("=" * 64)
    print(f"  Figuras salvas em: {OUTPUT_DIR}")
    print("=" * 64)
