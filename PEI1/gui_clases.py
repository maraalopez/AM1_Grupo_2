# gui_reactive.py
# ---------------------------------------------------------------------------
#  Cuadro de mando: TELEMETRÍA Y DINÁMICA DE ACTITUD SATELITAL
#  Teorema del eje intermedio (efecto Dzhanibekov)
#
#  Paneles:
#    1. Espacio de fases 3D (ω1, ω2, ω3): elipsoide de energía, elipsoide de
#       momento angular, separatriz y trayectoria numérica (con marcador vivo).
#    2. Actitud 3D del satélite en el sistema inercial (animación en vivo):
#       cuerpo + paneles solares, triedro de ejes principales, vector L (fijo
#       en el espacio inercial) y vector ω.
#    3. Deriva de energía ΔT(t) y de momento angular ΔL(t), con los tres
#       integradores superpuestos para poder contrastarlos.
# ---------------------------------------------------------------------------
import math
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import fisica as f
import esquemas_temporales as et

# ============================== TEMA ======================================
BG = '#0a0e17'      # fondo de la ventana
PANEL = '#111827'   # fondo de los gráficos
CARD = '#0f1623'    # fondo de las tarjetas de control
EDGE = '#26324a'    # bordes y rejillas
TEXT = '#d5dcea'
MUTED = '#7b8aa5'

C_ENERGY = '#ff5a5f'
C_MOMENT = '#4aa8ff'
C_SEPAR = '#a3e635'
C_AXES = ('#ff6b6b', '#69db7c', '#4dabf7')   # ejes 1, 2, 3

plt.style.use('dark_background')
plt.rcParams.update({
    'font.family': 'monospace',
    'font.size': 9,
    'figure.facecolor': BG,
    'savefig.facecolor': BG,
    'axes.facecolor': PANEL,
    'axes.edgecolor': EDGE,
    'axes.labelcolor': TEXT,
    'text.color': TEXT,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'grid.color': EDGE,
    'grid.alpha': 0.7,
})

# nombre -> (función esquema, color)
SCHEMES = {
    'Euler': (et.euler, '#ff6b6b'),
    'RK4': (et.rk4, '#4dd0e1'),
    'Crank-Nicolson': (et.crank_nicholson, '#b388ff'),
}

NMESH = 28            # resolución de las superficies de Poinsot (baja = más fluido)
FLOOR = 1e-17         # suelo para la escala logarítmica de la deriva
ATT_LIM = 1.45        # semilado del cubo del panel de actitud


# ======================= MATEMÁTICA DE ACTITUD ============================
def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def quat_between(a, b):
    """Cuaternión (w,x,y,z) que rota el vector a hacia el vector b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    d = float(a @ b)
    if d < -1 + 1e-12:                       # vectores opuestos
        ax = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, [0.0, 1.0, 0.0])
        ax /= np.linalg.norm(ax)
        return (0.0, ax[0], ax[1], ax[2])
    c = np.cross(a, b)
    q = np.array([1.0 + d, c[0], c[1], c[2]])
    q /= np.linalg.norm(q)
    return tuple(q)


def integrate_attitude(t, U, q0):
    """
    Integra la actitud a partir de ω(t) (en ejes cuerpo):  q' = 1/2 q ⊗ (0, ω).
    Se usa la exponencial de la rotación en cada paso (conserva |q| = 1).
    Devuelve q (w,x,y,z) de cuerpo -> inercial para cada instante.
    """
    n = len(t)
    Q = np.empty((n, 4))
    q = tuple(q0)
    Q[0] = q
    Ul, tl = U.tolist(), t.tolist()
    for k in range(n - 1):
        h = tl[k + 1] - tl[k]
        wx = 0.5 * (Ul[k][0] + Ul[k + 1][0])
        wy = 0.5 * (Ul[k][1] + Ul[k + 1][1])
        wz = 0.5 * (Ul[k][2] + Ul[k + 1][2])
        nw = math.sqrt(wx * wx + wy * wy + wz * wz)
        th = nw * h
        if th > 1e-15:
            s = math.sin(0.5 * th) / nw
            q = quat_mul(q, (math.cos(0.5 * th), s * wx, s * wy, s * wz))
            nq = math.sqrt(sum(c * c for c in q))
            q = tuple(c / nq for c in q)
        Q[k + 1] = q
    return Q


def quats_to_R(Q):
    w, x, y, z = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    R = np.empty((len(Q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


# ========================= GEOMETRÍA DEL SATÉLITE =========================
_BOX_IDX = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
            [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]       # -z +z -y +y +x -x
_BOX_N = np.array([[0, 0, -1], [0, 0, 1], [0, -1, 0],
                   [0, 1, 0], [1, 0, 0], [-1, 0, 0]], float)
_CORNERS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
LIGHT = np.array([0.35, -0.45, 0.82])
LIGHT /= np.linalg.norm(LIGHT)


def _box(center, half):
    pts = _CORNERS * np.asarray(half) + np.asarray(center)
    return pts[_BOX_IDX]                                    # (6, 4, 3)


def build_satellite(I):
    """
    Cuerpo = cuboide homogéneo con esos momentos de inercia
    (a² ∝ -I1+I2+I3, ...), acotado para que siempre sea visible.
    Los paneles solares son decorativos y van a lo largo del eje 2.
    """
    I1, I2, I3 = I
    d = np.sqrt(np.maximum([-I1 + I2 + I3, I1 - I2 + I3, I1 + I2 - I3], 0.0))
    d = np.clip(d / max(d.max(), 1e-12), 0.22, 1.0)
    h = 0.42 * d

    body_cols = [to_rgb(C_AXES[2]), to_rgb(C_AXES[2]), to_rgb(C_AXES[1]),
                 to_rgb(C_AXES[1]), to_rgb(C_AXES[0]), to_rgb(C_AXES[0])]
    body_cols = [np.array(c) * (1.0 if i % 2 else 0.62)      # caras - más oscuras
                 for i, c in enumerate(body_cols)]
    faces, cols, norms = [_box(np.zeros(3), h)], [np.array(body_cols)], [_BOX_N]

    panel_cols = np.array([to_rgb('#2563eb'), to_rgb('#3b82f6'),
                           to_rgb('#1e3a8a'), to_rgb('#1e3a8a'),
                           to_rgb('#1e3a8a'), to_rgb('#1e3a8a')])
    for s in (-1, 1):
        c = np.array([0.0, s * (h[1] + 0.05 + 0.45), 0.0])
        faces.append(_box(c, (0.30, 0.45, 0.012)))
        cols.append(panel_cols)
        norms.append(_BOX_N)
    return np.concatenate(faces), np.concatenate(cols), np.concatenate(norms)


# ============================ UTILIDADES GUI ==============================
def _style2d(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(EDGE)
    ax.grid(True, color=EDGE, linestyle=':', alpha=0.8)
    ax.tick_params(labelsize=8)


def _style3d(ax):
    ax.set_facecolor(PANEL)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.tick_params(labelsize=7, colors=MUTED)


def _make_slider(fig, rect, label, vmin, vmax, vinit, color, fmt='%.2f'):
    ax = fig.add_axes(rect, facecolor=EDGE)
    try:
        s = Slider(ax, label, vmin, vmax, valinit=vinit, valfmt=fmt, color=color,
                   initcolor='none', track_color=EDGE,
                   handle_style=dict(facecolor=color, edgecolor='white', size=9))
    except TypeError:                      # matplotlib antiguo
        s = Slider(ax, label, vmin, vmax, valinit=vinit, valfmt=fmt, color=color)
    s.label.set_color(TEXT)
    s.label.set_fontsize(9)
    s.valtext.set_color(color)
    s.valtext.set_fontsize(9)
    return s


def _make_button(fig, rect, text, color=EDGE, hover='#34456a'):
    ax = fig.add_axes(rect)
    b = Button(ax, text, color=color, hovercolor=hover)
    b.label.set_color(TEXT)
    b.label.set_fontsize(8)
    return b


def _clean_widget_axes(ax):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_visible(False)


# ============================== GUI =======================================
def lanzar_gui():
    fig = plt.figure(figsize=(17.5, 10.2), facecolor=BG)
    try:
        fig.canvas.manager.set_window_title('Telemetría y dinámica de actitud satelital')
    except Exception:
        pass

    # ---------------- fondo con tarjetas y títulos ----------------
    bg = fig.add_axes([0, 0, 1, 1], zorder=-10)
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)
    bg.set_autoscale_on(False)      # que las coordenadas de bg.text sigan siendo de figura
    bg.set_axis_off()
    bg.set_navigate(False)

    fig.text(0.03, 0.962, 'TELEMETRÍA Y DINÁMICA DE ACTITUD SATELITAL',
             fontsize=18, fontweight='bold', color='white', va='center')
    fig.text(0.03, 0.930,
             'Teorema del eje intermedio  ·  Ecuaciones de Euler  ·  Efecto Dzhanibekov',
             fontsize=9.5, color=MUTED, va='center')

    def card(x, y, w, h, title, color):
        bg.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle='round,pad=0,rounding_size=0.008',
                                    fc=CARD, ec=EDGE, lw=1.0, transform=bg.transAxes))
        bg.text(x + 0.012, y + h - 0.028, title, color=color, fontsize=9,
                fontweight='bold', va='center')
        bg.plot([x + 0.012, x + w - 0.012], [y + h - 0.047] * 2, color=EDGE, lw=1)

    CY, CH = 0.015, 0.30
    card(0.020, CY, 0.215, CH, 'PARÁMETROS FÍSICOS · INERCIAS', '#fbbf24')
    card(0.245, CY, 0.215, CH, 'CONDICIONES INICIALES · ω(0)', '#2dd4bf')
    card(0.470, CY, 0.270, CH, 'MOTOR NUMÉRICO', '#b388ff')
    card(0.750, CY, 0.230, CH, 'ANIMACIÓN', '#f472b6')
    bg.text(0.5, 0.004, 'ESPACIO: play/pausa   ·   arrastra los paneles 3D para rotar la vista',
            color=MUTED, fontsize=7.5, ha='center', va='center')

    # ---------------- paneles de visualización ----------------
    gs = fig.add_gridspec(2, 3, left=0.03, right=0.985, top=0.885, bottom=0.345,
                          wspace=0.12, hspace=0.18, width_ratios=[1.2, 1.2, 1.0])
    ax_ph = fig.add_subplot(gs[:, 0], projection='3d')
    ax_att = fig.add_subplot(gs[:, 1], projection='3d')
    ax_T = fig.add_subplot(gs[0, 2])
    ax_L = fig.add_subplot(gs[1, 2], sharex=ax_T)

    # --- espacio de fases ---
    _style3d(ax_ph)
    ax_ph.set_title('ESPACIO DE FASES  (ω₁, ω₂, ω₃)', loc='left', fontsize=11,
                    fontweight='bold', color='white', pad=6)
    ax_ph.set_xlabel(r'$\omega_1$', labelpad=-2)
    ax_ph.set_ylabel(r'$\omega_2$', labelpad=-2)
    ax_ph.set_zlabel(r'$\omega_3$', labelpad=-2)
    ax_ph.set_box_aspect((1, 1, 1), zoom=1.2)
    ax_ph.view_init(elev=22, azim=-58)
    traj_full, = ax_ph.plot([], [], [], color='#e2e8f0', lw=1.0, alpha=0.35)
    traj_done, = ax_ph.plot([], [], [], color='white', lw=2.2)
    pt_ph, = ax_ph.plot([], [], [], 'o', color='#ffd43b', ms=8, mec='white', mew=1.2)
    ax_ph.legend(handles=[
        Line2D([0], [0], color=C_ENERGY, lw=6, alpha=0.4, label='Elipsoide de energía'),
        Line2D([0], [0], color=C_MOMENT, lw=2, alpha=0.8, label='Elipsoide de momento L'),
        Line2D([0], [0], color=C_SEPAR, lw=2.5, ls='--', label='Separatriz'),
        Line2D([0], [0], color='white', lw=2, label='ω(t) numérico'),
    ], loc='upper left', fontsize=7.5, frameon=False)

    # --- actitud del satélite ---
    _style3d(ax_att)
    ax_att.set_title('ACTITUD DEL SATÉLITE  (sistema inercial)', loc='left', fontsize=11,
                     fontweight='bold', color='white', pad=6)
    ax_att.set_xlim(-ATT_LIM, ATT_LIM)
    ax_att.set_ylim(-ATT_LIM, ATT_LIM)
    ax_att.set_zlim(-ATT_LIM, ATT_LIM)
    ax_att.set_box_aspect((1, 1, 1), zoom=1.25)
    ax_att.set_xticklabels([])
    ax_att.set_yticklabels([])
    ax_att.set_zticklabels([])
    ax_att.tick_params(length=0)
    ax_att.view_init(elev=20, azim=-55)

    sat = Poly3DCollection([np.zeros((4, 3))], edgecolor='#0b1220', linewidths=0.6)
    ax_att.add_collection3d(sat)
    axis_lines = [ax_att.plot([], [], [], color=C_AXES[i], lw=2.4,
                              label=f'eje principal {i + 1}')[0] for i in range(3)]
    line_L, = ax_att.plot([], [], [], color='#ffd43b', lw=2.6, ls='--',
                          label='L  (fijo en el espacio)')
    line_w, = ax_att.plot([], [], [], color='white', lw=2.2, label='ω')
    trail_att, = ax_att.plot([], [], [], color='#f472b6', lw=1.4, alpha=0.8,
                             label='traza del eje de giro inicial')
    ax_att.legend(loc='upper right', fontsize=7.5, frameon=False)
    chip = ax_att.text2D(0.02, 0.97, '', transform=ax_att.transAxes, fontsize=9,
                         fontweight='bold', va='top', linespacing=1.5)
    readout = ax_att.text2D(0.02, 0.02, '', transform=ax_att.transAxes, fontsize=8,
                            color=TEXT, va='bottom')

    # --- derivas ---
    for ax in (ax_T, ax_L):
        _style2d(ax)
    ax_T.set_title('DERIVA DE ENERGÍA  ΔT(t)', loc='left', fontsize=11,
                   fontweight='bold', color=C_ENERGY, pad=6)
    ax_L.set_title('DERIVA DE MOMENTO  ΔL(t)', loc='left', fontsize=11,
                   fontweight='bold', color=C_MOMENT, pad=6)
    ax_T.set_ylabel('error absoluto')
    ax_L.set_ylabel('error absoluto')
    ax_L.set_xlabel('Tiempo [s]')
    ax_T.tick_params(labelbottom=False)
    lines_T, lines_L = {}, {}
    for name, (_, col) in SCHEMES.items():
        lines_T[name], = ax_T.plot([], [], color=col, lw=1.1, label=name)
        lines_L[name], = ax_L.plot([], [], color=col, lw=1.1, label=name)
    leg = ax_T.legend(handles=list(lines_T.values()), loc='upper left',
                      fontsize=8, frameon=False)
    for tx in leg.get_texts():
        tx.set_color(TEXT)
    cur_T = ax_T.axvline(0, color='white', lw=1, alpha=0.7)
    cur_L = ax_L.axvline(0, color='white', lw=1, alpha=0.7)

    # ---------------- controles ----------------
    H = 0.026
    R1, R2, R3 = 0.215, 0.160, 0.105

    # tarjeta 1: inercias
    x = 0.020
    s_I1 = _make_slider(fig, [x + 0.045, R1, 0.125, H], r'$I_1$', 0.1, 5.0, 1.0, '#fbbf24')
    s_I2 = _make_slider(fig, [x + 0.045, R2, 0.125, H], r'$I_2$', 0.1, 5.0, 2.0, '#fbbf24')
    s_I3 = _make_slider(fig, [x + 0.045, R3, 0.125, H], r'$I_3$', 0.1, 5.0, 3.0, '#fbbf24')
    info = bg.text(x + 0.012, 0.075, '', color=MUTED, fontsize=8, va='center',
                   linespacing=1.6)

    # tarjeta 2: ω(0)
    x = 0.245
    s_w1 = _make_slider(fig, [x + 0.065, R1, 0.100, H], r'$\omega_1(0)$', -5.0, 5.0, 0.01, '#2dd4bf')
    s_w2 = _make_slider(fig, [x + 0.065, R2, 0.100, H], r'$\omega_2(0)$', -5.0, 5.0, 1.0, '#2dd4bf')
    s_w3 = _make_slider(fig, [x + 0.065, R3, 0.100, H], r'$\omega_3(0)$', -5.0, 5.0, 0.01, '#2dd4bf')
    b_min = _make_button(fig, [x + 0.012, 0.035, 0.064, 0.045], 'eje menor')
    b_mid = _make_button(fig, [x + 0.080, 0.035, 0.064, 0.045], 'eje interm.',
                         color='#5b2a3a', hover='#8a3b52')
    b_max = _make_button(fig, [x + 0.148, 0.035, 0.056, 0.045], 'eje mayor')

    # tarjeta 3: motor numérico
    x = 0.470
    s_dt = _make_slider(fig, [x + 0.045, R1, 0.170, H], r'$\Delta t$', 0.002, 0.2, 0.05,
                        '#b388ff', fmt='%.3f')
    s_tf = _make_slider(fig, [x + 0.045, R2, 0.170, H], r'$t_f$', 5.0, 60.0, 20.0,
                        '#b388ff', fmt='%.0f s')
    ax_radio = fig.add_axes([x + 0.010, 0.035, 0.115, 0.105])
    _clean_widget_axes(ax_radio)
    radio = RadioButtons(ax_radio, tuple(SCHEMES), active=2, activecolor='#b388ff')
    for lab, (_, col) in zip(radio.labels, SCHEMES.values()):
        lab.set_color(col)
        lab.set_fontsize(9)
    ax_chk = fig.add_axes([x + 0.140, 0.035, 0.125, 0.105])
    _clean_widget_axes(ax_chk)
    checks = CheckButtons(ax_chk, ['Comparar todos', 'Deriva en log'], [True, True])
    for lab in checks.labels:
        lab.set_color(TEXT)
        lab.set_fontsize(8.5)

    # tarjeta 4: animación
    x = 0.750
    b_play = _make_button(fig, [x + 0.015, 0.200, 0.095, 0.050], '||  Pausa',
                          color='#7a2150', hover='#a22d6c')
    b_reset = _make_button(fig, [x + 0.120, 0.200, 0.095, 0.050], '↺  Reiniciar')
    s_speed = _make_slider(fig, [x + 0.055, R2, 0.130, H], 'vel.', 0.2, 6.0, 2.0,
                           '#f472b6', fmt='%.1fx')
    status = bg.text(x + 0.015, 0.075, '', color=MUTED, fontsize=8, va='center',
                     linespacing=1.6)

    # ---------------- estado ----------------
    st = dict(dirty=True, last_change=0.0, key=None, cache={}, res=None,
              playing=True, tsim=0.0, last_tick=time.perf_counter(),
              surf=[], sep=[], fills=[], sat=None, lim=1.0, params=None)

    # ---------------- lógica ----------------
    def read_params():
        I = (s_I1.val, s_I2.val, s_I3.val)
        w0 = (s_w1.val, s_w2.val, s_w3.val)
        return I, w0, s_dt.val, s_tf.val

    def update_info():
        I = np.array([s_I1.val, s_I2.val, s_I3.val])
        o = np.argsort(I)
        tri = all(I[i] <= I[j] + I[k] + 1e-12
                  for i, j, k in ((0, 1, 2), (1, 0, 2), (2, 0, 1)))
        info.set_text(f'menor = I{o[0] + 1}   interm. = I{o[1] + 1}   mayor = I{o[2] + 1}\n'
                      f'desig. triangular: {"OK" if tri else "VIOLADA (cuerpo no realizable)"}')

    def mark(*_):
        st['dirty'] = True
        st['last_change'] = time.perf_counter()
        update_info()
        fig.canvas.draw_idle()

    def simulate(name, I, w0, dt, tf):
        n = max(int(tf / dt), 2)
        t = np.asarray(et.partition(0, tf, n), float)
        U0 = np.array(w0, float)
        Il = list(I)

        def F(omega, t_val):
            return f.euler_equations(omega, Il, t_val)

        with np.errstate(all='ignore'):
            U = np.asarray(et.cauchy_problem(F=F, U0=U0, t=t, scheme=SCHEMES[name][0]), float)
            dT = np.asarray(et.calcular_deriva(U, f.invariant_energy, Il), float)
            dL = np.asarray(et.calcular_deriva(U, f.invariant_angular_momentum, Il), float)
        bad = ~np.isfinite(U).all(axis=1) | (np.abs(U) > 1e6).any(axis=1)
        diverged = None
        if bad.any():                                  # p. ej. Euler con Δt grande
            cut = max(int(bad.argmax()), 2)
            diverged = t[cut - 1]
            t, U, dT, dL = t[:cut], U[:cut], dT[:cut], dL[:cut]
        dT = np.nan_to_num(dT, nan=np.inf)
        dL = np.nan_to_num(dL, nan=np.inf)
        return dict(t=t, U=U, dT=dT, dL=dL, diverged=diverged)

    def ensure_attitude(res, I, w0):
        if 'Rb' in res:
            return
        U, t = res['U'], res['t']
        Ib = np.array(I)
        Lb0 = Ib * np.array(w0)
        q0 = quat_between(Lb0, [0, 0, 1]) if np.linalg.norm(Lb0) > 1e-12 else (1, 0, 0, 0)
        Q = integrate_attitude(t, U, q0)
        res['Rb'] = quats_to_R(Q)
        dom = int(np.argmax(np.abs(w0)))
        res['dom'] = dom
        res['tips'] = res['Rb'][:, :, dom] * 1.15
        wmax = max(np.linalg.norm(U, axis=1).max(), 1e-12)
        res['wscale'] = 1.1 / wmax
        res['Lscale'] = 1.35 / max(np.linalg.norm(Lb0), 1e-12)

    def update_phase(I, w0):
        for a in st['surf'] + st['sep']:
            try:
                a.remove()
            except Exception:
                pass
        st['surf'], st['sep'] = [], []
        Il = list(I)
        U0 = np.array(w0, float)
        T0 = f.invariant_energy(U0, Il)
        L0 = f.invariant_angular_momentum(U0, Il)
        mT, mL = f.superficies_poinsot(Il, T0, np.sqrt(L0), num_puntos=NMESH)
        st['surf'].append(ax_ph.plot_surface(mT[0], mT[1], mT[2], color=C_ENERGY,
                                             alpha=0.13, linewidth=0, shade=False,
                                             antialiased=False))
        st['surf'].append(ax_ph.plot_wireframe(mL[0], mL[1], mL[2], color=C_MOMENT,
                                               alpha=0.38, linewidth=0.6,
                                               rstride=2, cstride=2))
        sep = f.calcular_separatriz(Il, np.linalg.norm(U0))
        branches = sep if isinstance(sep, (list, tuple)) else [sep]
        for br in branches:
            br = np.asarray(br)
            st['sep'].extend(ax_ph.plot(br[:, 0], br[:, 1], br[:, 2], color=C_SEPAR,
                                        lw=3, ls='--'))
        lim = 1.15 * max(np.nanmax(np.abs(m)) for m in (*mT, *mL))
        lim = max(lim, 1e-3)
        ax_ph.set_xlim(-lim, lim)
        ax_ph.set_ylim(-lim, lim)
        ax_ph.set_zlim(-lim, lim)

    def apply_scale():
        log = checks.get_status()[1]
        for ax in (ax_T, ax_L):
            ax.set_yscale('log' if log else 'linear')

    def update_drift(sel, tf):
        for c in st['fills']:
            try:
                c.remove()
            except Exception:
                pass
        st['fills'] = []
        compare, log = checks.get_status()
        apply_scale()
        for name, (_, col) in SCHEMES.items():
            for ax, lines, key in ((ax_T, lines_T, 'dT'), (ax_L, lines_L, 'dL')):
                ln = lines[name]
                r = st['cache'].get(name)
                if r is None or not (compare or name == sel):
                    ln.set_visible(False)
                    continue
                y = np.maximum(r[key], FLOOR) if log else r[key]
                y = np.where(np.isfinite(y), y, np.nan)
                is_sel = name == sel
                ln.set_data(r['t'], y)
                ln.set_visible(True)
                ln.set_linewidth(2.4 if is_sel else 1.1)
                ln.set_alpha(1.0 if is_sel else 0.55)
                ln.set_zorder(3 if is_sel else 2)
                if is_sel:
                    base = FLOOR if log else 0.0
                    st['fills'].append(ax.fill_between(r['t'], base, np.nan_to_num(y, nan=base),
                                                       color=col, alpha=0.12, linewidth=0))
        ax_T.set_xlim(0, tf)
        for ax in (ax_T, ax_L):
            ax.relim(visible_only=True)
            ax.autoscale_view(scalex=False)

    def draw_frame(k):
        res = st['res']
        if res is None:
            return
        I, w0 = st['params'][0], st['params'][1]
        k = int(min(max(k, 0), len(res['t']) - 1))
        t, U, Rb = res['t'], res['U'], res['Rb']
        R = Rb[k]

        # satélite
        verts = st['sat'][0] @ R.T
        shade = 0.40 + 0.60 * np.clip((st['sat'][2] @ R.T) @ LIGHT, 0, 1)
        rgb = np.clip(st['sat'][1] * shade[:, None], 0, 1)
        sat.set_verts(list(verts))
        sat.set_facecolor(np.c_[rgb, np.full(len(rgb), 0.97)])

        for i in range(3):
            tip = R[:, i] * 1.15
            axis_lines[i].set_data_3d([0, tip[0]], [0, tip[1]], [0, tip[2]])
        Li = R @ (np.array(I) * U[k]) * res['Lscale']
        line_L.set_data_3d([0, Li[0]], [0, Li[1]], [0, Li[2]])
        wi = R @ U[k] * res['wscale']
        line_w.set_data_3d([0, wi[0]], [0, wi[1]], [0, wi[2]])
        s0 = max(0, k - 600)
        step = max(1, (k - s0) // 300)
        tp = res['tips'][s0:k + 1:step]
        trail_att.set_data_3d(tp[:, 0], tp[:, 1], tp[:, 2])

        # espacio de fases
        step = max(1, k // 1000)
        traj_done.set_data_3d(U[:k + 1:step, 0], U[:k + 1:step, 1], U[:k + 1:step, 2])
        pt_ph.set_data_3d([U[k, 0]], [U[k, 1]], [U[k, 2]])

        # cursores y telemetría
        cur_T.set_xdata([t[k], t[k]])
        cur_L.set_xdata([t[k], t[k]])
        readout.set_text(
            f't  = {t[k]:6.2f} s      paso {k}/{len(t) - 1}\n'
            f'ω  = ({U[k, 0]:+.3f}, {U[k, 1]:+.3f}, {U[k, 2]:+.3f})\n'
            f'ΔT = {res["dT"][k]:.2e}     ΔL = {res["dL"][k]:.2e}')

    def recompute():
        st['dirty'] = False
        I, w0, dt, tf = read_params()
        st['params'] = (I, w0, dt, tf)
        if np.linalg.norm(w0) < 1e-9:
            status.set_text('ω(0) = 0  →  el satélite no gira')
            fig.canvas.draw_idle()
            return
        key = (I, w0, dt, tf)
        if key != st['key']:
            st['key'], st['cache'] = key, {}
            surf_needed = True
        else:
            surf_needed = False
        sel = radio.value_selected
        compare = checks.get_status()[0]
        t0 = time.perf_counter()
        try:
            for name in (list(SCHEMES) if compare else [sel]):
                if name not in st['cache']:
                    st['cache'][name] = simulate(name, I, w0, dt, tf)
            res = st['cache'][sel]
            if sel not in st['cache']:
                st['cache'][sel] = simulate(sel, I, w0, dt, tf)
                res = st['cache'][sel]
            ensure_attitude(res, I, w0)
            if surf_needed or not st['surf']:
                update_phase(I, w0)
        except Exception as exc:                    # no tumbar la GUI por un fallo numérico
            status.set_text(f'ERROR numérico: {exc}')
            status.set_color('#ff6b6b')
            fig.canvas.draw_idle()
            return

        st['res'] = res
        st['sat'] = build_satellite(I)
        # trayectoria completa
        U = res['U']
        traj_full.set_data_3d(U[:, 0], U[:, 1], U[:, 2])
        pt_ph.set_color(SCHEMES[sel][1])
        update_drift(sel, tf)

        # etiqueta de estabilidad según el eje de giro inicial
        dom = res['dom']
        rank = list(np.argsort(I)).index(dom)
        txt, col = (('EJE DE MENOR INERCIA → estable', '#4ade80'),
                    ('EJE INTERMEDIO → INESTABLE', '#ff6b6b'),
                    ('EJE DE MAYOR INERCIA → estable', '#4ade80'))[rank]
        chip.set_text(f'{txt}\ngiro inicial en el eje {dom + 1}')
        chip.set_color(col)

        msg = f'{len(res["t"]) - 1} pasos · {sel}\ncálculo: {time.perf_counter() - t0:.2f} s'
        if res['diverged'] is not None:
            msg += f'\n⚠ diverge en t ≈ {res["diverged"]:.1f} s'
            status.set_color('#fbbf24')
        else:
            status.set_color(MUTED)
        status.set_text(msg)

        st['tsim'] = 0.0
        draw_frame(0)
        fig.canvas.draw_idle()

    def tick():
        now = time.perf_counter()
        if st['dirty'] and now - st['last_change'] > 0.12:
            recompute()
            st['last_tick'] = time.perf_counter()
            return
        dt_real = min(now - st['last_tick'], 0.1)
        st['last_tick'] = now
        res = st['res']
        if res is None or not st['playing']:
            return
        st['tsim'] += dt_real * s_speed.val
        t = res['t']
        if st['tsim'] >= t[-1]:
            st['tsim'] = 0.0
        draw_frame(int(np.searchsorted(t, st['tsim'])))
        fig.canvas.draw_idle()

    # ---------------- callbacks ----------------
    for s in (s_I1, s_I2, s_I3, s_w1, s_w2, s_w3, s_dt, s_tf):
        s.on_changed(mark)
    radio.on_clicked(mark)

    def on_check(label):
        if label == 'Deriva en log':
            if st['res'] is not None:
                update_drift(radio.value_selected, st['params'][3])
            fig.canvas.draw_idle()
        else:
            mark()
    checks.on_clicked(on_check)

    def toggle_play(_=None):
        st['playing'] = not st['playing']
        b_play.label.set_text('||  Pausa' if st['playing'] else '▶  Reproducir')
        fig.canvas.draw_idle()

    def reset(_=None):
        st['tsim'] = 0.0
        draw_frame(0)
        fig.canvas.draw_idle()

    b_play.on_clicked(toggle_play)
    b_reset.on_clicked(reset)

    def preset(rank):
        def _cb(_):
            idx = int(np.argsort([s_I1.val, s_I2.val, s_I3.val])[rank])
            vals = [0.01, 0.01, 0.01]
            vals[idx] = 1.0
            for s, v in zip((s_w1, s_w2, s_w3), vals):
                s.eventson = False
                s.set_val(v)
                s.eventson = True
            mark()
        return _cb

    b_min.on_clicked(preset(0))
    b_mid.on_clicked(preset(1))
    b_max.on_clicked(preset(2))

    fig.canvas.mpl_connect('key_press_event',
                           lambda e: toggle_play() if e.key == ' ' else None)

    # ---------------- arranque ----------------
    update_info()
    recompute()
    timer = fig.canvas.new_timer(interval=40)
    timer.add_callback(tick)
    timer.start()

    # imprescindible: si no se guardan referencias, Python recoge los widgets
    fig._gui_refs = [s_I1, s_I2, s_I3, s_w1, s_w2, s_w3, s_dt, s_tf, s_speed, radio,
                     checks, b_play, b_reset, b_min, b_mid, b_max, timer]
    fig._gui_api = dict(recompute=recompute, tick=tick, draw_frame=draw_frame, state=st)
    plt.show()
    return fig


if __name__ == '__main__':
    lanzar_gui()