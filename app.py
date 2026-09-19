"""
Interaktivní vizualizátor Metody nejmenších čtverců (OLS) a Gradientního sestupu (GD).
Připraveno pro běh lokálně i online (Streamlit Community Cloud).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# Nastavení stránky
st.set_page_config(
    page_title="OLS & Gradientní sestup | Interaktivní simulátor",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stylizace a CSS pro moderní vzhled
st.markdown("""
<style>
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        font-family: monospace;
        color: #0f172a;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #94a3b8;
    }
    .optimal-badge {
        color: #059669;
    }
    .current-badge {
        color: #2563eb;
    }
    .gd-badge {
        color: #7c3aed;
    }
</style>
""", unsafe_allow_html=True)

# Výchozí datasety
PRESETS = {
    "Původní vzorová data (DEFAULT)": pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "y": [2.2, 2.8, 4.5, 3.8, 5.8, 5.2, 7.1]
    }),
    "Vliv odlehlé hodnoty (Outlier)": pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "y": [2.0, 2.7, 3.9, 4.8, 6.1, 1.2, 8.0]
    }),
    "Silná lineární závislost": pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "y": [1.9, 3.1, 4.0, 5.2, 6.0, 7.1, 7.9, 9.1]
    }),
    "Rozptýlená data s vysokým šumem": pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "y": [4.5, 2.1, 6.8, 3.2, 7.5, 4.9, 8.4, 6.2]
    })
}

# Inicializace session state
if "preset_choice" not in st.session_state:
    st.session_state.preset_choice = "Původní vzorová data (DEFAULT)"
if "points_df" not in st.session_state:
    st.session_state.points_df = PRESETS[st.session_state.preset_choice].copy()
if "slope" not in st.session_state:
    st.session_state.slope = 0.80
if "intercept" not in st.session_state:
    st.session_state.intercept = 1.20
if "iteration" not in st.session_state:
    st.session_state.iteration = 0
if "loss_history" not in st.session_state:
    st.session_state.loss_history = []
if "learning_rate" not in st.session_state:
    st.session_state.learning_rate = 0.02

# Funkce pro analytický výpočet OLS
def compute_ols(df: pd.DataFrame):
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    n = len(x)
    if n < 2:
        return 1.0, 0.0, 0.0, 0.0, 0.0

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    var_x = np.sum((x - mean_x) ** 2)
    cov_xy = np.sum((x - mean_x) * (y - mean_y))

    if var_x == 0:
        slope = 0.0
    else:
        slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x

    y_pred = slope * x + intercept
    sse = np.sum((y - y_pred) ** 2)
    tss = np.sum((y - mean_y) ** 2)
    r2 = 1.0 - (sse / tss) if tss > 0 else 1.0

    return slope, intercept, sse, tss, r2

# Funkce pro výpočet gradientů MSE
def compute_gradients(df: pd.DataFrame, slope: float, intercept: float):
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    n = len(x)
    if n == 0:
        return 0.0, 0.0, 0.0
    y_pred = slope * x + intercept
    error = y_pred - y
    d_b0 = (2.0 / n) * np.sum(error)
    d_b1 = (2.0 / n) * np.sum(error * x)
    mse = np.mean(error ** 2)
    return float(d_b0), float(d_b1), float(mse)

# Krok gradientního sestupu
def step_gradient_descent(num_steps=1):
    x = st.session_state.points_df["x"].to_numpy()
    y = st.session_state.points_df["y"].to_numpy()
    n = len(x)
    if n == 0:
        return

    lr = st.session_state.learning_rate
    b0 = st.session_state.intercept
    b1 = st.session_state.slope

    # Pokud historie začíná, zaznamenáme počáteční stav
    if len(st.session_state.loss_history) == 0:
        init_mse = np.mean(((b1 * x + b0) - y) ** 2)
        st.session_state.loss_history.append(float(init_mse))

    for _ in range(num_steps):
        y_pred = b1 * x + b0
        error = y_pred - y
        d_b0 = (2.0 / n) * np.sum(error)
        d_b1 = (2.0 / n) * np.sum(error * x)

        b0 -= lr * d_b0
        b1 -= lr * d_b1
        st.session_state.iteration += 1
        new_mse = np.mean(((b1 * x + b0) - y) ** 2)
        st.session_state.loss_history.append(float(new_mse))

    st.session_state.intercept = float(np.round(b0, 4))
    st.session_state.slope = float(np.round(b1, 4))

# Výpočet optimálních parametrů pro aktuální data
opt_slope, opt_intercept, opt_sse, tss, opt_r2 = compute_ols(st.session_state.points_df)
n_points = len(st.session_state.points_df)
opt_mse = (opt_sse / n_points) if n_points > 0 else 0.0

# Aktuální gradienty
grad_b0, grad_b1, current_mse = compute_gradients(
    st.session_state.points_df, 
    st.session_state.slope, 
    st.session_state.intercept
)

# Callbacky pro tlačítka
def snap_to_optimal():
    st.session_state.slope = float(np.round(opt_slope, 3))
    st.session_state.intercept = float(np.round(opt_intercept, 3))

def reset_to_default():
    st.session_state.slope = 0.80
    st.session_state.intercept = 1.20
    st.session_state.iteration = 0
    st.session_state.loss_history = []

def reset_gd_only():
    st.session_state.iteration = 0
    st.session_state.loss_history = []

def on_preset_change():
    selected = st.session_state.preset_choice
    if selected in PRESETS:
        st.session_state.points_df = PRESETS[selected].copy()
        s, i, _, _, _ = compute_ols(st.session_state.points_df)
        st.session_state.slope = float(np.round(s, 2))
        st.session_state.intercept = float(np.round(i, 2))
        st.session_state.iteration = 0
        st.session_state.loss_history = []

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Ovládání modelu")
    
    st.selectbox(
        "Přednastavený dataset",
        options=list(PRESETS.keys()) + ["Vlastní data (editovat v tabulce)"],
        key="preset_choice",
        on_change=on_preset_change
    )
    
    st.divider()
    st.subheader("Parametry regresní přímky")
    st.caption(r"Rovnice: $\hat{y} = \beta_1 x + \beta_0$")

    # Dynamické přizpůsobení mezí posuvníků podle dat a aktuální hodnoty
    slider_min_slope = min(-3.0, float(np.floor(min(st.session_state.slope, opt_slope) - 1.0)))
    slider_max_slope = max(4.0, float(np.ceil(max(st.session_state.slope, opt_slope) + 1.0)))
    slider_min_intercept = min(-5.0, float(np.floor(min(st.session_state.intercept, opt_intercept) - 2.0)))
    slider_max_intercept = max(10.0, float(np.ceil(max(st.session_state.intercept, opt_intercept) + 2.0)))

    # Posuvníky (řízené přes key v session_state)
    st.slider(
        "Směrnice (Slope, β₁)",
        min_value=slider_min_slope,
        max_value=slider_max_slope,
        step=0.02,
        key="slope",
        help="Určuje sklon regresní přímky."
    )

    st.slider(
        "Posun (Intercept, β₀)",
        min_value=slider_min_intercept,
        max_value=slider_max_intercept,
        step=0.05,
        key="intercept",
        help="Hodnota, kde přímka protíná osu y."
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.button("🎯 Nastavit OLS", on_click=snap_to_optimal, width="stretch", help="Nastaví analyticky optimální směrnici i posun.")
    with col_btn2:
        st.button("🔄 Reset", on_click=reset_to_default, width="stretch", help="Vrátí výchozí hodnoty (0.80, 1.20).")

    # ==================== SEKCE: GRADIENTNÍ SESTUP ====================
    st.divider()
    st.subheader("⚡ Gradientní sestup")
    st.caption("Iterativní učení váhových parametrů minimalizací MSE.")

    st.slider(
        "Rychlost učení (Learning Rate, α)",
        min_value=0.001,
        max_value=0.100,
        step=0.005,
        key="learning_rate",
        help="Velikost kroku v záporném směru gradientu."
    )

    # Zobrazení stavu GD
    col_gd1, col_gd2 = st.columns(2)
    with col_gd1:
        st.metric("Iterace", st.session_state.iteration)
    with col_gd2:
        st.metric("Aktuální MSE", f"{current_mse:.4f}")

    st.caption(f"Gradienty: ∂MSE/∂β₀ = `{grad_b0:+.3f}`, ∂MSE/∂β₁ = `{grad_b1:+.3f}`")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.button("▶️ +1", on_click=lambda: step_gradient_descent(1), width="stretch", help="Provést 1 krok")
    with col_s2:
        st.button("⏩ +10", on_click=lambda: step_gradient_descent(10), width="stretch", help="Provést 10 kroků")
    with col_s3:
        st.button("⚡ +30", on_click=lambda: step_gradient_descent(30), width="stretch", help="Provést 30 kroků")
    with col_s4:
        st.button("🚀 +100", on_click=lambda: step_gradient_descent(100), width="stretch", help="Provést 100 kroků")

    if st.button("↺ Vynulovat iterace GD", on_click=reset_gd_only, width="stretch"):
        pass

    st.divider()
    st.subheader("Možnosti zobrazení")
    show_squares = st.checkbox("Zobrazit čtverce chyb (eᵢ²)", value=True, help="Vykreslí geometrický čtverec nad každou vertikální odchylkou.")
    show_residuals = st.checkbox("Zobrazit rezidua (svislé čáry)", value=True, help="Vykreslí úsečky od bodu k přímce.")
    show_optimal_line = st.checkbox("Zobrazit optimální OLS přímku (reference)", value=False, help="Zelená čárkovaná přímka reprezentující globální minimum OLS.")
    lock_aspect = st.checkbox("Zamknout poměr stran 1:1", value=True, help="Zaručuje, že čtverce budou mít na obrazovce skutečně tvar čtverce.")

    st.divider()
    with st.expander("📂 Import vlastních dat (CSV)"):
        uploaded_file = st.file_uploader("Nahrajte CSV soubor se sloupci 'x' a 'y'", type=["csv"])
        if uploaded_file is not None:
            try:
                custom_df = pd.read_csv(uploaded_file)
                if "x" in custom_df.columns and "y" in custom_df.columns:
                    st.session_state.points_df = custom_df[["x", "y"]].dropna().astype(float)
                    st.session_state.iteration = 0
                    st.session_state.loss_history = []
                    st.success(f"Načteno {len(st.session_state.points_df)} bodů!")
                    st.rerun()
                else:
                    st.error("CSV musí obsahovat sloupce s názvy 'x' a 'y'.")
            except Exception as e:
                st.error(f"Chyba při načítání: {e}")

# ==================== HLAVNÍ OBSAH ====================

st.title("📐 Metoda nejmenších čtverců & Gradientní sestup")
st.markdown("Interaktivní simulace lineární regrese: srovnejte **analytické řešení (OLS)** s **iterativní optimalizací (Gradient Descent)**.")

# Příprava dat a výpočtů pro aktuální přímku
df = st.session_state.points_df.copy()
cur_slope = st.session_state.slope
cur_intercept = st.session_state.intercept

df["y_pred"] = cur_slope * df["x"] + cur_intercept
df["residual"] = df["y"] - df["y_pred"]
df["sq_error"] = df["residual"] ** 2

current_sse = df["sq_error"].sum()
current_r2 = 1.0 - (current_sse / tss) if tss > 0 else 1.0
delta_sse = current_sse - opt_sse

# Karty metrik nahoře
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">AKTUÁLNÍ SSE (Součet čtverců)</div>
        <div class="metric-value current-badge">{current_sse:.3f}</div>
        <div class="metric-sub">MSE = {current_mse:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">OPTIMÁLNÍ OLS MINIMUM</div>
        <div class="metric-value optimal-badge">{opt_sse:.3f}</div>
        <div class="metric-sub">Globální minimum OLS</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    diff_text = f"+{delta_sse:.3f}" if delta_sse > 0.0001 else "0.000 (Optimum!)"
    diff_color = "#059669" if delta_sse <= 0.001 else "#dc2626"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">ODCHYLKA OD OPTIMA (Δ SSE)</div>
        <div class="metric-value" style="color: {diff_color};">{diff_text}</div>
        <div class="metric-sub">O kolik jste horší než OLS</div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">STAV GRADIENTNÍHO SESTUPU</div>
        <div class="metric-value gd-badge">{st.session_state.iteration} kroků</div>
        <div class="metric-sub">α = {st.session_state.learning_rate:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

# Vykreslení grafu Plotly
x_min_data = df["x"].min()
x_max_data = df["x"].max()
y_min_data = df["y"].min()
y_max_data = df["y"].max()

# Výpočet rozsahů os s rezervou
x_margin = max(1.0, (x_max_data - x_min_data) * 0.25)
y_margin = max(1.0, (y_max_data - y_min_data) * 0.35)

x_plot_min = max(0.0, float(np.floor(x_min_data - x_margin)))
x_plot_max = float(np.ceil(x_max_data + x_margin + 1.0))
y_plot_min = float(np.floor(min(0.0, y_min_data - y_margin)))
y_plot_max = float(np.ceil(y_max_data + y_margin + 1.0))

fig = go.Figure()

# 1. Geometrické čtverce chyb
if show_squares:
    for idx, row in df.iterrows():
        x_pt = row["x"]
        y_pt = row["y"]
        y_pr = row["y_pred"]
        res = row["residual"]
        abs_res = abs(res)

        # Čtverec kreslíme směrem doprava od svislice bodu
        rect_x0 = x_pt
        rect_x1 = x_pt + abs_res
        rect_y0 = min(y_pt, y_pr)
        rect_y1 = max(y_pt, y_pr)

        fig.add_shape(
            type="rect",
            x0=rect_x0,
            y0=rect_y0,
            x1=rect_x1,
            y1=rect_y1,
            fillcolor="rgba(239, 68, 68, 0.20)",
            line=dict(color="#ef4444", width=1.2, dash="dot"),
            layer="below"
        )

# 2. Svislé úsečky reziduí
if show_residuals:
    for idx, row in df.iterrows():
        fig.add_shape(
            type="line",
            x0=row["x"],
            y0=row["y"],
            x1=row["x"],
            y1=row["y_pred"],
            line=dict(color="#ef4444", width=2.5),
            layer="above"
        )

# 3. Optimální OLS přímka (volitelná reference)
if show_optimal_line:
    x_line = np.array([x_plot_min, x_plot_max])
    y_opt_line = opt_slope * x_line + opt_intercept
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_opt_line,
        mode="lines",
        name=f"Optimum OLS: ŷ = {opt_slope:.2f}x + {opt_intercept:.2f}",
        line=dict(color="#10b981", width=2.5, dash="dash"),
        hoverinfo="skip"
    ))

# 4. Aktuální modelová regresní přímka
x_line = np.array([x_plot_min, x_plot_max])
y_cur_line = cur_slope * x_line + cur_intercept
fig.add_trace(go.Scatter(
    x=x_line,
    y=y_cur_line,
    mode="lines",
    name=f"Model: ŷ = {cur_slope:.2f}x + {cur_intercept:.2f}",
    line=dict(color="#2563eb", width=3.5),
    hoverinfo="skip"
))

# 5. Datové body
fig.add_trace(go.Scatter(
    x=df["x"],
    y=df["y"],
    mode="markers",
    name="Datové body",
    marker=dict(
        size=11,
        color="#0f172a",
        line=dict(width=2, color="#ffffff")
    ),
    customdata=np.stack((df["y_pred"], df["residual"], df["sq_error"]), axis=-1),
    hovertemplate=(
        "<b>Bod (x=%{x:.2f}, y=%{y:.2f})</b><br>" +
        "Predikce ŷ: %{customdata[0]:.2f}<br>" +
        "Reziduum e: %{customdata[1]:.2f}<br>" +
        "Čtverec chyby e²: %{customdata[2]:.3f}<extra></extra>"
    )
))

# Konfigurace vzhledu grafu
layout_kwargs = dict(
    height=540,
    margin=dict(l=40, r=40, t=30, b=40),
    plot_bgcolor="#f8fafc",
    paper_bgcolor="#ffffff",
    xaxis=dict(
        range=[x_plot_min, x_plot_max],
        zeroline=True,
        zerolinecolor="#cbd5e1",
        gridcolor="#f1f5f9",
        title="x",
        dtick=1
    ),
    yaxis=dict(
        range=[y_plot_min, y_plot_max],
        zeroline=True,
        zerolinecolor="#cbd5e1",
        gridcolor="#f1f5f9",
        title="y",
        dtick=1
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    hovermode="closest"
)

if lock_aspect:
    layout_kwargs["yaxis"]["scaleanchor"] = "x"
    layout_kwargs["yaxis"]["scaleratio"] = 1.0

fig.update_layout(**layout_kwargs)

st.plotly_chart(fig, width="stretch")

# Sekce: Tabulky, Graf konvergence a Teorie
tab_loss, tab_table, tab_math, tab_deploy = st.tabs([
    "📉 Konvergence ztráty (Loss Curve)",
    "📊 Tabulka bodů a reziduí", 
    "📐 Matematika: OLS vs. Gradient Descent", 
    "🚀 Jak aplikaci nasadit online"
])

with tab_loss:
    st.subheader("Vývoj ztrátové funkce (MSE) v čase")
    st.caption("Sledujte, jak Gradient Descent postupně konverguje k teoretickému minimu OLS.")

    if len(st.session_state.loss_history) > 1:
        loss_df = pd.DataFrame({
            "Iterace": list(range(len(st.session_state.loss_history))),
            "MSE": st.session_state.loss_history
        })

        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=loss_df["Iterace"],
            y=loss_df["MSE"],
            mode="lines+markers",
            name="MSE Gradient Descent",
            line=dict(color="#7c3aed", width=2.5),
            marker=dict(size=5)
        ))
        fig_loss.add_trace(go.Scatter(
            x=[0, len(st.session_state.loss_history) - 1],
            y=[opt_mse, opt_mse],
            mode="lines",
            name=f"Analytické OLS minimum ({opt_mse:.4f})",
            line=dict(color="#10b981", width=2, dash="dash")
        ))
        fig_loss.update_layout(
            height=380,
            margin=dict(l=40, r=40, t=25, b=40),
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#ffffff",
            xaxis=dict(title="Číslo iterace (k)", gridcolor="#f1f5f9"),
            yaxis=dict(title="Mean Squared Error (MSE)", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_loss, width="stretch")
    else:
        st.info("Zatím jste neprovedli žádné kroky gradientního sestupu. Klikněte v postranním panelu na **▶️ 1 krok**, **⏩ 10 kroků** nebo **🎬 Animovat**, abyste viděli křivku učení!")

with tab_table:
    st.subheader("Detailní rozpis reziduí jednotlivých bodů")
    st.caption("Můžete v tabulce přímo upravit hodnoty nebo přidat nové body!")
    
    edited_df = st.data_editor(
        df[["x", "y"]],
        num_rows="dynamic",
        width="stretch",
        key="data_editor"
    )
    
    # Pokud uživatel změnil data v tabulce, aktualizujeme
    if not edited_df.equals(st.session_state.points_df[["x", "y"]]):
        st.session_state.points_df = edited_df.dropna().astype(float)
        st.session_state.iteration = 0
        st.session_state.loss_history = []
        st.rerun()

    # Zobrazení výsledků reziduí
    display_df = df.copy()
    display_df = display_df.rename(columns={
        "x": "x",
        "y": "y (skutečnost)",
        "y_pred": "ŷ (predikce)",
        "residual": "e = y - ŷ (reziduum)",
        "sq_error": "e² (plocha čtverce)"
    })
    st.dataframe(
        display_df.style.format({
            "x": "{:.2f}",
            "y (skutečnost)": "{:.2f}",
            "ŷ (predikce)": "{:.2f}",
            "e = y - ŷ (reziduum)": "{:+.3f}",
            "e² (plocha čtverce)": "{:.4f}"
        }),
        width="stretch"
    )

with tab_math:
    st.subheader("Dva přístupy k lineární regresi")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(r"""
        ### 1. Analytické OLS (Normal Equations)
        Minimalizuje SSE jedním přímým výpočtem:
        $$SSE(\beta_0, \beta_1) = \sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i)^2$$
        
        Položením parciálních derivací rovno nule získáme globální minimum:
        $$\hat{\beta}_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2} = \frac{\operatorname{Cov}(X, Y)}{\operatorname{Var}(X)}$$
        $$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

        - ✅ **Výhoda:** Okamžité, přesné řešení v jediném kroku.
        - ⚠️ **Nevýhoda:** Pro matici s $d$ dimenzemi vyžaduje inverzi $(X^T X)^{-1}$ se složitostí $\mathcal{O}(d^3)$. Pro obrovské datasety je nepoužitelné.
        """)
    with col_t2:
        st.markdown(r"""
        ### 2. Gradientní sestup (Gradient Descent)
        Iterativní optimalizace ztrátové funkce MSE:
        $$MSE = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2 = \frac{1}{n} \sum_{i=1}^n (\beta_1 x_i + \beta_0 - y_i)^2$$

        Parciální derivace (gradienty):
        $$\frac{\partial MSE}{\partial \beta_0} = \frac{2}{n} \sum_{i=1}^n (\hat{y}_i - y_i)$$
        $$\frac{\partial MSE}{\partial \beta_1} = \frac{2}{n} \sum_{i=1}^n (\hat{y}_i - y_i) x_i$$

        Pravidlo aktualizace vah s rychlostí učení $\alpha$:
        $$\beta_0 \leftarrow \beta_0 - \alpha \frac{\partial MSE}{\partial \beta_0}$$
        $$\beta_1 \leftarrow \beta_1 - \alpha \frac{\partial MSE}{\partial \beta_1}$$

        - ✅ **Výhoda:** Snadno škáluje na miliardy datových bodů a miliony parametrů (základ hlubokého učení).
        - ⚠️ **Nevýhoda:** Vyžaduje ladění hyperparametru $\alpha$ (při příliš velkém $\alpha$ diverguje).
        """)

with tab_deploy:
    st.subheader("Jak nasadit tuto aplikaci online zdarma")
    st.markdown("""
    Tato aplikace je kompletně připravena pro bezplatný hosting na **Streamlit Community Cloud** nebo **Hugging Face Spaces**.

    #### Možnost 1: Streamlit Community Cloud (Nejjednodušší, 2 minuty)
    1. Vytvořte si bezplatný účet na [GitHub.com](https://github.com) (pokud nemáte).
    2. Vytvořte nový GitHub repozitář (např. `ols-regression-app`) a nahrajte do něj tyto soubory:
       - `app.py`
       - `requirements.txt`
       - `.streamlit/config.toml`
       - `README.md`
    3. Přejděte na [share.streamlit.io](https://share.streamlit.io) a přihlaste se přes GitHub.
    4. Klikněte na **"New app"**, vyberte váš repozitář a hlavní soubor `app.py`.
    5. Klikněte na **"Deploy!"** – během 1 minuty dostanete veřejnou URL, kterou můžete poslat komukoliv.

    #### Možnost 2: Spuštění lokálně
    V terminálu stačí spustit:
    ```bash
    streamlit run app.py
    ```
    """)
