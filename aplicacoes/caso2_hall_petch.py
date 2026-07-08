#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Caso 2 — Previsão de Propriedades Mecânicas via Hall-Petch
===========================================================
Conecta a evolução do tamanho de grão com a resistência mecânica por
meio da relação de Hall-Petch:

    σ_y = σ₀ + K_HP / √d       [MPa, d em mm]

Materiais modelados:
  ● Aço de baixo carbono (IF steel): σ₀ = 70 MPa,  K_HP = 15,8 MPa·mm⁰·⁵
  ● Alumínio série 1xxx (puro):      σ₀ = 15 MPa,  K_HP =  3,2 MPa·mm⁰·⁵

Escala física: 1 px = 5 µm  |  grade 200×200 px ≡ 1,0 mm²

Correção estereológica 2D→3D (Wicksell 1925)
───────────────────────────────────────────────
  O diâmetro médio extraído da grade é o de uma SEÇÃO PLANAR 2D aleatória,
  que subestima sistematicamente o diâmetro 3D verdadeiro dos grãos: para
  uma seção aleatória de esferas, o diâmetro médio da seção é 2/3 do
  diâmetro 3D (problema do corpúsculo de Wicksell). Logo:
      D_3D = D_2D / (2/3) = 1,5 * D_2D
  Aplicamos esse fator (STEREO_FACTOR) antes de usar o diâmetro nas
  equações de Hall-Petch (que pressupõem constantes de materiais 3D reais).
  O diâmetro 2D bruto é mantido no painel 1 (saída direta da simulação).

Resultado esperado:
  O gráfico mostra que o recozimento (crescimento de grão) REDUZ a
  tensão de escoamento — troca resistência por ductilidade. Este é
  o compromisso fundamental do tratamento térmico de recozimento.

Referências:
  Hall-Petch: Hall (1951) Proc. Phys. Soc. B 64:747; Petch (1953) JISI 174:25
  Parâmetros de aço: Gladman, T. (1997) The Physical Metallurgy of Microalloyed
    Steels. The Institute of Materials, London.
  Correção estereológica: Wicksell, S.D. (1925) Biometrika 17(1-2):84-99.
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Localiza o motor CA no diretório pai ────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # aplicacoes/
PARENT    = os.path.dirname(_THIS_DIR)                   # simulacao_ca_classica/
sys.path.insert(0, PARENT)
import simulacao_ca_classica as ca

# ── Parâmetros de simulação ──────────────────────────────────────────────────
GRID_SIZE    = 200
N_NUCLEI     = 600
N_STEPS      = 1500
RECORD_EVERY = 50
SEED         = 42

# ── Calibração física ────────────────────────────────────────────────────────
PX_TO_UM   = 5.0    # 1 pixel = 5 µm
K_CA_REF   = 0.232  # px²/CAS (T_NOISE = 0,5)
K_PHYS_REF = 3.0    # µm²/s  a 700 °C (aço)
CAS_TO_MIN = K_CA_REF * PX_TO_UM**2 / (K_PHYS_REF * 60)  # min/CAS ≈ 0,032 min/CAS

# Fator de correção estereológica 2D→3D (Wicksell 1925): diâmetro médio de
# seção planar aleatória = 2/3 do diâmetro 3D verdadeiro (grãos ~esféricos)
STEREO_FACTOR = 1.5

# ── Parâmetros de Hall-Petch (d em mm) ──────────────────────────────────────
# σ_y = σ₀ + K_HP / √d,  d em mm,  σ em MPa
MATERIALS = {
    'Aço de baixo carbono\n(σ₀=70, K=15,8)': {
        'sigma_0': 70.0,
        'K_HP':    15.8,      # MPa·mm⁰·⁵  (≈ 0,5 MPa·m⁰·⁵)
        'color':   'steelblue',
        'marker':  'o',
    },
    'Alumínio 1xxx\n(σ₀=15, K=3,2)': {
        'sigma_0': 15.0,
        'K_HP':    3.2,       # MPa·mm⁰·⁵  (≈ 0,1 MPa·m⁰·⁵)
        'color':   'darkorange',
        'marker':  's',
    },
}

OUTPUT_DIR = os.path.join(_THIS_DIR, 'resultados')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def hall_petch(d_mm, sigma_0, K_HP):
    return sigma_0 + K_HP / np.sqrt(d_mm)


def main():
    print("=" * 64)
    print("  CASO 2 — Previsão de Resistência Mecânica via Hall-Petch")
    print(f"  Grade: {GRID_SIZE}×{GRID_SIZE}  |  T_NOISE = 0,5  |  {N_STEPS} passos")
    print("=" * 64)

    ca.T_NOISE   = 0.5
    ca.P_NEUTRAL = 0.5
    np.random.seed(SEED)

    grid = ca.initialize_grid(GRID_SIZE, N_NUCLEI, SEED)
    s0   = ca.grain_statistics(grid)

    steps_rec  = [0]
    R_rec      = [s0['mean_radius']]
    n_gr_rec   = [s0['n_grains']]

    for step in range(1, N_STEPS + 1):
        grid = ca.ca_step(grid)
        if step % RECORD_EVERY == 0:
            s = ca.grain_statistics(grid)
            steps_rec.append(step)
            R_rec.append(s['mean_radius'])
            n_gr_rec.append(s['n_grains'])

    steps_arr = np.array(steps_rec, dtype=float)
    R_px      = np.array(R_rec)
    d_um      = 2.0 * R_px * PX_TO_UM         # µm (seção 2D, saída bruta da simulação)
    d_mm      = d_um / 1000.0                  # mm
    d_um_3d   = d_um * STEREO_FACTOR          # µm (corrigido p/ diâmetro 3D, Wicksell 1925)
    d_mm_3d   = d_mm * STEREO_FACTOR          # mm
    t_min     = steps_arr * CAS_TO_MIN         # minutos

    print(f"\n  Diâmetro 2D (bruto):      {d_um[0]:.1f} µm  →  {d_um[-1]:.1f} µm")
    print(f"  Diâmetro 3D (corrigido):  {d_um_3d[0]:.1f} µm  →  {d_um_3d[-1]:.1f} µm"
          f"  (fator Wicksell = {STEREO_FACTOR})")

    for name, mat in MATERIALS.items():
        sy = hall_petch(d_mm_3d, mat['sigma_0'], mat['K_HP'])
        lbl = name.replace('\n', ' ')
        print(f"\n  {lbl}:")
        print(f"    σ_y inicial: {sy[0]:.1f} MPa  (d_3D = {d_um_3d[0]:.1f} µm)")
        print(f"    σ_y final:   {sy[-1]:.1f} MPa  (d_3D = {d_um_3d[-1]:.1f} µm)")
        print(f"    Redução:     {sy[0] - sy[-1]:.1f} MPa  ({(sy[0]-sy[-1])/sy[0]*100:.1f} %)")

    # ── Figura: 3 painéis ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Painel 1 — Diâmetro de grão vs. tempo (valor 2D bruto da simulação)
    ax = axes[0]
    ax.plot(t_min, d_um, '-o', ms=3, lw=2, color='teal', label='Simulação CA (seção 2D)')
    ax.set_xlabel('Tempo de recozimento (min)', fontsize=11)
    ax.set_ylabel('Diâmetro médio d  (µm)', fontsize=11)
    ax.set_title('Crescimento de Grão\nvs. Tempo de Recozimento (seção 2D bruta)', fontsize=11)
    ax.grid(True, alpha=0.3)

    ax_top = ax.twiny()
    cas_ticks = np.arange(0, N_STEPS + 1, 300)
    ax_top.set_xlim(0, N_STEPS * CAS_TO_MIN)
    ax_top.set_xticks(cas_ticks * CAS_TO_MIN)
    ax_top.set_xticklabels([str(c) for c in cas_ticks], fontsize=8)
    ax_top.set_xlabel('Passos CA (CAS)', fontsize=9)

    # Painel 2 — σ_y vs. tempo (diâmetro corrigido para 3D, Wicksell 1925)
    ax = axes[1]
    for name, mat in MATERIALS.items():
        sy = hall_petch(d_mm_3d, mat['sigma_0'], mat['K_HP'])
        ax.plot(t_min, sy, '-', ms=3, lw=2,
                color=mat['color'], marker=mat['marker'], label=name)

    ax.set_xlabel('Tempo de recozimento (min)', fontsize=11)
    ax.set_ylabel('Tensão de escoamento σ_y  (MPa)', fontsize=11)
    ax.set_title('Hall-Petch: σ_y × Tempo\n(d corrigido p/ 3D; recozimento amolece o material)',
                 fontsize=11)
    ax.legend(fontsize=8.5, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Painel 3 — Curva de Hall-Petch clássica + trajetória da simulação
    # (eixo de diâmetro já corrigido para 3D via fator de Wicksell)
    ax = axes[2]
    d_range_mm = np.linspace(max(d_mm_3d.min() * 0.5, 0.001), d_mm_3d.max() * 1.5, 300)
    d_range_um = d_range_mm * 1000

    for name, mat in MATERIALS.items():
        sy_curve = hall_petch(d_range_mm, mat['sigma_0'], mat['K_HP'])
        ax.plot(d_range_um, sy_curve, '--', lw=1.5,
                color=mat['color'], alpha=0.55, label=f'{name} (curva teórica)')
        sy_sim = hall_petch(d_mm_3d, mat['sigma_0'], mat['K_HP'])
        sc = ax.scatter(d_um_3d, sy_sim, c=t_min, cmap='viridis', s=35,
                        zorder=5, edgecolors='none')

    cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label('Tempo (min)', fontsize=9)

    ax.set_xlabel('Diâmetro médio 3D corrigido d  (µm)', fontsize=11)
    ax.set_ylabel('Tensão de escoamento σ_y  (MPa)', fontsize=11)
    ax.set_title('Curva de Hall-Petch\n(pontos = trajetória do recozimento, d 3D corrigido)',
                 fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Caso 2 — Previsão de Propriedades Mecânicas via Hall-Petch\n'
        f'(escala: 1 px = {PX_TO_UM} µm  |  T_NOISE = 0,5  ~  700 °C  |  '
        f'correção estereológica 2D→3D ×{STEREO_FACTOR}, Wicksell 1925)',
        fontsize=11)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, 'caso2_hall_petch.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\n  Figura salva: {out}")

    print("\n  INTERPRETAÇÃO DOS RESULTADOS:")
    print("  O painel central mostra o 'custo mecânico' do recozimento:")
    print("    quanto mais tempo/temperatura, menor a resistência do material.")
    print("  O painel direito mostra a trajetória no espaço Hall-Petch:")
    print("    cada ponto é uma snapshot da simulação, colorido pelo tempo.")
    print(f"  O diâmetro 2D bruto da grade foi corrigido por ×{STEREO_FACTOR}")
    print("    (Wicksell 1925) antes de entrar nas equações de Hall-Petch, que")
    print("    pressupõem o diâmetro 3D verdadeiro dos grãos.")
    print("  Para aplicações reais: calibrar PX_TO_UM e K_PHYS_REF com")
    print("    dados experimentais do material e temperatura específicos.")


if __name__ == '__main__':
    main()
