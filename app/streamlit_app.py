from __future__ import annotations

import os
import pandas as pd
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Checkout Guidance",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# GLOBAL CSS — design premium
# =========================================================

st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── App background ── */
.stApp {
    background: #f4f6f9;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #1a1f2e !important;
    border-right: none;
}
section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] .stSelectbox label {
    color: #94a3b8 !important;
    font-size: 11px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e8ecf0;
    border-radius: 16px;
    padding: 20px 24px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
div[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-size: 28px !important;
    font-weight: 800 !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ── Section title helper ── */
h2, h3 { color: #0f172a !important; font-weight: 700 !important; }

/* ── Hide streamlit default elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA
# =========================================================

CSV_PATH = "outputs/checkout_predictions.csv"

if not os.path.exists(CSV_PATH):
    st.error("⚠️  Données introuvables. Lancez d'abord `python main.py`.")
    st.stop()

df = pd.read_csv(CSV_PATH)


# =========================================================
# HELPERS
# =========================================================

def compute_checkout_view(data: pd.DataFrame) -> pd.DataFrame:
    view = (
        data.groupby(["store_id", "checkout_id"], as_index=False)
        .agg(
            total_transactions=("transaction_id", "count"),
            avg_duration_sec=("pred_transaction_duration_sec", "mean"),
            avg_abandon_risk=("pred_abandon_proba", "mean"),
            avg_congestion=("congestion_score_norm", "mean"),
            max_congestion=("congestion_score_norm", "max"),
        )
    )
    view["estimated_wait_min"] = (
        view["avg_duration_sec"] / 60 + view["avg_congestion"] / 25
    ).round(1)

    def status(wait: float) -> str:
        if wait >= 6:   return "critical"
        if wait >= 4:   return "high"
        if wait >= 2:   return "medium"
        return "low"

    view["status"] = view["estimated_wait_min"].apply(status)
    return view


STATUS_META = {
    "low":      {"label": "Disponible",   "color": "#10b981", "bg": "#ecfdf5", "dot": "🟢"},
    "medium":   {"label": "Modérée",      "color": "#f59e0b", "bg": "#fffbeb", "dot": "🟡"},
    "high":     {"label": "Chargée",      "color": "#ef4444", "bg": "#fef2f2", "dot": "🔴"},
    "critical": {"label": "Critique",     "color": "#7c3aed", "bg": "#f5f3ff", "dot": "🔴"},
}

def wait_color(wait: float) -> str:
    if wait >= 6: return "#7c3aed"
    if wait >= 4: return "#ef4444"
    if wait >= 2: return "#f59e0b"
    return "#10b981"

def customer_message(wait: float) -> str:
    if wait >= 6: return "Caisse la plus rapide disponible en ce moment"
    if wait >= 4: return "Temps d'attente modéré — merci de vous y diriger"
    if wait >= 2: return "Caisse recommandée pour réduire votre attente"
    return "Caisse disponible immédiatement — aucune attente"


# =========================================================
# SIDEBAR — Filtres
# =========================================================

with st.sidebar:
    st.markdown("""
    <div style="padding: 24px 8px 8px 8px;">
        <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;
                    color:#475569;font-weight:600;margin-bottom:6px;">
            CHECKOUT GUIDANCE
        </div>
        <div style="font-size:22px;font-weight:800;color:#f1f5f9;margin-bottom:4px;">
            Dashboard
        </div>
        <div style="height:2px;background:linear-gradient(90deg,#3b82f6,transparent);
                    border-radius:2px;margin-bottom:28px;"></div>
    </div>
    """, unsafe_allow_html=True)

    stores = ["All"] + sorted(df["store_id"].unique().tolist())
    selected_store = st.selectbox("Magasin", stores)

    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

    # Mini légende
    st.markdown("""
    <div style="padding:16px;background:#111827;border-radius:12px;">
        <div style="font-size:10px;letter-spacing:1.5px;color:#475569;
                    text-transform:uppercase;font-weight:600;margin-bottom:12px;">
            LÉGENDE
        </div>
    """, unsafe_allow_html=True)

    for key, meta in STATUS_META.items():
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <div style="width:10px;height:10px;border-radius:50%;
                        background:{meta['color']};flex-shrink:0;"></div>
            <span style="font-size:12px;color:#94a3b8;font-weight:500;">
                {meta['label']}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# FILTER DATA
# =========================================================

filtered = df[df["store_id"] == selected_store].copy() if selected_store != "All" else df.copy()
checkout_view = compute_checkout_view(filtered)

if checkout_view.empty:
    st.warning("Aucune donnée disponible pour ce magasin.")
    st.stop()

best = checkout_view.sort_values(
    ["estimated_wait_min", "avg_abandon_risk"], ascending=[True, True]
).iloc[0]

best_checkout   = best["checkout_id"]
best_wait       = float(best["estimated_wait_min"])
best_color      = wait_color(best_wait)
best_num        = best_checkout.replace("CHK_", "")
best_status     = best["status"]
best_status_meta= STATUS_META[best_status]

recommended = checkout_view.sort_values(
    ["estimated_wait_min", "avg_abandon_risk"], ascending=[True, True]
).head(8).copy()


# =========================================================
# HEADER
# =========================================================

store_label = selected_store if selected_store != "All" else "Tous les magasins"

st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            margin-bottom:24px;">
    <div>
        <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;
                    color:#64748b;font-weight:600;">PASSAGE EN CAISSE</div>
        <div style="font-size:28px;font-weight:800;color:#0f172a;margin-top:2px;">
            Orientation clients
        </div>
    </div>
    <div style="text-align:right;">
        <div style="font-size:12px;color:#94a3b8;font-weight:500;">Magasin sélectionné</div>
        <div style="font-size:18px;font-weight:700;color:#1e40af;">{store_label}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# HERO — Caisse recommandée
# =========================================================

# Pre-compute values to avoid expressions inside f-string HTML
_abandon_pct  = round(float(best["avg_abandon_risk"]) * 100, 1)
_best_tx      = int(best["total_transactions"])
_best_cong    = round(float(best["avg_congestion"]), 2)
_best_msg     = customer_message(best_wait)
_best_lbl     = best_status_meta["label"].upper()
_best_dot_col = best_status_meta["color"]

_hero = (
    f'<div style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 60%,#1e3a5f 100%);'
    f'border-radius:24px;padding:40px 48px;margin-bottom:20px;'
    f'box-shadow:0 20px 60px rgba(15,23,42,0.18);">'
    f'<div style="display:flex;align-items:center;gap:48px;">'
    f'<div style="width:160px;height:160px;border-radius:28px;background:{best_color};'
    f'display:flex;flex-direction:column;align-items:center;justify-content:center;'
    f'flex-shrink:0;box-shadow:0 12px 40px {best_color}55;">'
    f'<div style="font-size:13px;font-weight:600;color:rgba(255,255,255,0.75);'
    f'letter-spacing:2px;text-transform:uppercase;margin-bottom:4px;">CAISSE</div>'
    f'<div style="font-size:72px;font-weight:900;color:white;line-height:1;">{best_num}</div>'
    f'</div>'
    f'<div style="flex:1;">'
    f'<div style="display:inline-flex;align-items:center;gap:6px;'
    f'background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);'
    f'border-radius:20px;padding:4px 14px;margin-bottom:16px;">'
    f'<div style="width:7px;height:7px;border-radius:50%;background:{_best_dot_col};"></div>'
    f'<span style="font-size:12px;color:rgba(255,255,255,0.7);font-weight:500;">{_best_lbl}</span>'
    f'</div>'
    f'<div style="font-size:42px;font-weight:800;color:white;line-height:1.1;margin-bottom:12px;">{best_wait} min</div>'
    f'<div style="font-size:16px;color:#94a3b8;margin-bottom:24px;max-width:380px;">{_best_msg}</div>'
    f'<div style="display:flex;gap:32px;">'
    f'<div><div style="font-size:11px;color:#475569;font-weight:600;letter-spacing:1px;text-transform:uppercase;">Risque abandon</div>'
    f'<div style="font-size:20px;font-weight:700;color:#e2e8f0;margin-top:2px;">{_abandon_pct}%</div></div>'
    f'<div><div style="font-size:11px;color:#475569;font-weight:600;letter-spacing:1px;text-transform:uppercase;">Transactions</div>'
    f'<div style="font-size:20px;font-weight:700;color:#e2e8f0;margin-top:2px;">{_best_tx}</div></div>'
    f'<div><div style="font-size:11px;color:#475569;font-weight:600;letter-spacing:1px;text-transform:uppercase;">Congestion</div>'
    f'<div style="font-size:20px;font-weight:700;color:#e2e8f0;margin-top:2px;">{_best_cong}</div></div>'
    f'</div></div></div></div>'
)
st.markdown(_hero, unsafe_allow_html=True)



# =========================================================
# AUTRES CAISSES RECOMMANDÉES
# =========================================================

st.markdown("""
<div style="font-size:14px;font-weight:700;color:#0f172a;
            letter-spacing:0.3px;margin-bottom:14px;">
    Autres caisses disponibles
</div>
""", unsafe_allow_html=True)

cols = st.columns(min(len(recommended), 8))

for col, (_, row) in zip(cols, recommended.iterrows()):
    num    = row["checkout_id"].replace("CHK_", "")
    wait   = float(row["estimated_wait_min"])
    color  = wait_color(wait)
    status = row["status"]
    smeta  = STATUS_META[status]

    with col:
        st.markdown(f"""
        <div style="
            background:white;
            border:1.5px solid #e2e8f0;
            border-radius:16px;
            padding:16px 12px;
            text-align:center;
            box-shadow:0 1px 4px rgba(0,0,0,0.04);
            transition:all 0.2s;
        ">
            <div style="
                width:52px; height:52px; border-radius:14px;
                background:{color};
                display:flex; align-items:center; justify-content:center;
                margin:0 auto 10px auto;
                box-shadow: 0 4px 12px {color}44;
            ">
                <span style="font-size:22px;font-weight:900;color:white;">{num}</span>
            </div>
            <div style="font-size:17px;font-weight:800;color:#0f172a;">{wait} min</div>
            <div style="
                font-size:10px;font-weight:600;color:{smeta['color']};
                letter-spacing:0.5px;text-transform:uppercase;margin-top:4px;
            ">{smeta['label']}</div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# MÉTRIQUES
# =========================================================

st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Transactions analysées", f"{len(filtered):,}")
m2.metric("Caisse recommandée", best_checkout)
m3.metric("Attente moyenne", f"{round(checkout_view['estimated_wait_min'].mean(), 1)} min")
m4.metric("Caisses critiques", int((checkout_view["status"] == "critical").sum()))


# =========================================================
# GRAPHIQUES — 2 colonnes
# =========================================================

st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
left, right = st.columns(2)

with left:
    st.markdown("""
    <div style="font-size:14px;font-weight:700;color:#0f172a;margin-bottom:12px;">
        Répartition des niveaux de congestion
    </div>
    """, unsafe_allow_html=True)

    congestion_counts = (
        filtered["congestion_level"]
        .value_counts()
        .reindex(["low", "medium", "high", "critical"], fill_value=0)
        .rename(index={"low":"Faible","medium":"Modérée","high":"Élevée","critical":"Critique"})
    )
    st.bar_chart(congestion_counts, color="#3b82f6")

with right:
    st.markdown("""
    <div style="font-size:14px;font-weight:700;color:#0f172a;margin-bottom:12px;">
        Attente estimée par caisse
    </div>
    """, unsafe_allow_html=True)

    wait_chart = (
        checkout_view
        .sort_values("estimated_wait_min", ascending=False)
        .set_index("checkout_id")[["estimated_wait_min"]]
        .rename(columns={"estimated_wait_min": "Attente (min)"})
    )
    st.bar_chart(wait_chart, color="#f59e0b")


# =========================================================
# TABLE — Pilotage caisses
# =========================================================

st.markdown("""
<div style="font-size:14px;font-weight:700;color:#0f172a;margin:24px 0 12px 0;">
    Tableau de pilotage des caisses
</div>
""", unsafe_allow_html=True)

# Badge de statut dans le tableau
def status_badge(status: str) -> str:
    m = STATUS_META.get(status, STATUS_META["low"])
    return f"{m['dot']} {m['label']}"

ops = checkout_view.copy()
ops["status_display"] = ops["status"].apply(status_badge)
ops["avg_abandon_risk"] = (ops["avg_abandon_risk"] * 100).round(1).astype(str) + "%"
ops["avg_duration_sec"]  = ops["avg_duration_sec"].round(0).astype(int).astype(str) + "s"
ops["avg_congestion"]    = ops["avg_congestion"].round(2)
ops["max_congestion"]    = ops["max_congestion"].round(2)

ops_display = ops.rename(columns={
    "checkout_id":         "Caisse",
    "total_transactions":  "Transactions",
    "avg_duration_sec":    "Durée moy.",
    "avg_abandon_risk":    "Risque abandon",
    "avg_congestion":      "Congestion moy.",
    "max_congestion":      "Congestion max",
    "estimated_wait_min":  "Attente (min)",
    "status_display":      "Statut",
})[["Caisse","Transactions","Durée moy.","Risque abandon",
    "Congestion moy.","Attente (min)","Statut"]]

st.dataframe(
    ops_display.sort_values("Attente (min)", ascending=True),
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# TABLE — Transactions sensibles
# =========================================================

st.markdown("""
<div style="font-size:14px;font-weight:700;color:#0f172a;margin:24px 0 12px 0;">
    Transactions à risque élevé
</div>
""", unsafe_allow_html=True)

tx_cols = [
    "transaction_id", "checkout_id", "transaction_hour",
    "nb_items", "nb_scans_retries", "nb_help_requests",
    "pred_transaction_duration_sec", "pred_abandon_proba",
    "congestion_score_norm", "congestion_level", "recommendation",
]
available_cols = [c for c in tx_cols if c in filtered.columns]

st.dataframe(
    filtered.sort_values(
        ["congestion_score_norm", "pred_abandon_proba"],
        ascending=False,
    )[available_cols].head(20),
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div style="
    margin-top:40px;
    padding:16px 0;
    border-top:1px solid #e2e8f0;
    text-align:center;
    color:#94a3b8;
    font-size:12px;
">
    Données synthétiques — Démonstration produit &nbsp;·&nbsp;
    Checkout Guidance System
</div>
""", unsafe_allow_html=True)