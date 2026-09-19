# 📐 Interaktivní vizualizátor Metody nejmenších čtverců (OLS) & Gradientní sestup

Tato aplikace umožňuje interaktivně zkoumat princip **Metody nejmenších čtverců (Ordinary Least Squares - OLS)** a **Gradientního sestupu (Gradient Descent)** v lineární regresi.

Tento repozitář je navržen pro **duální nasazení (Dual Deployment)** – podporuje v jednom projektu jak **Streamlit Cloud**, tak **Vercel**:

| Vlastnost | Case A: Streamlit Cloud | Case B: Vercel (Edge Web App) |
| :--- | :--- | :--- |
| **Vstupní soubor** | `app.py` (+ `requirements.txt`) | `index.html` (+ `vercel.json`) |
| **Běhové prostředí** | Python 3.10+ (Streamlit server) | Statický Global Edge CDN (Vanilla JS + SVG) |
| **Silné stránky** | Data science ekosystém, Plotly, editovatelná tabulka dat | Okamžité načtení (0s cold start), 60 FPS animace v reálném čase |
| **Cena hostingu** | Zdarma (Streamlit Community Cloud) | Zdarma (Vercel Hobby) |

---

## 🚀 Jak aplikaci spustit lokálně

### 1. Varianta: Streamlit
```bash
pip install -r requirements.txt
streamlit run app.py
```
Aplikace se otevře na adrese `http://localhost:8501`.

### 2. Varianta: Samostatná webová verze (Vercel / Browser)
Dvakrát klikněte na soubor `index.html` v libovolném prohlížeči – funguje okamžitě bez instalace čehokoliv.

---

## 🌐 Jak aplikaci nasadit online (ZDARMA)

### 🔴 Case 1: Streamlit Community Cloud
1. Přejděte na [share.streamlit.io](https://share.streamlit.io) a přihlaste se přes GitHub.
2. Klikněte na **"New app"**:
   - **Repository:** `Kilitar/OLS_Regresion_Tool`
   - **Branch:** `main`
   - **Main file path:** `app.py`
3. Klikněte na **"Deploy!"** – získáte stálou adresu `https://<vase-jmeno>.streamlit.app`.

### ▲ Case 2: Vercel
1. Přejděte na [vercel.com](https://vercel.com) a přihlaste se přes GitHub.
2. Klikněte na **"Add New..."** ➔ **"Project"**.
3. Vyberte repozitář `OLS_Regresion_Tool` a klikněte na **"Import"**.
4. Vercel automaticky načte konfigurační soubor `vercel.json` – není potřeba nic nastavovat.
5. Klikněte na **"Deploy"** – během 10 sekund získáte bleskovou adresu na celosvětové CDN síti (např. `https://ols-regresion-tool.vercel.app`).

---

## 📊 Funkce aplikace

- **Geometrické čtverce reziduí ($e_i^2$)**: Vizuální ukázka toho, proč se metoda nazývá metoda nejmenších *čtverců*.
- **Posuvníky směrnice ($\beta_1$) a posunu ($\beta_0$)**: Manuální hledání optimální přímky s ochranou proti přetečení.
- **Tlačítko "Nastavit optimální OLS přímku"**: Okamžitě dosadí analyticky spočtené globální minimum.
- **Gradientní sestup (Gradient Descent)**:
  - Volba rychlosti učení $\alpha$ (Learning Rate).
  - Tlačítka kroků: `+1`, `+10`, `+30`, `+100`.
  - Výpočet parciálních derivací $\frac{\partial\text{MSE}}{\partial\beta_0}$ a $\frac{\partial\text{MSE}}{\partial\beta_1}$ v reálném čase.
  - Ochrana proti výbuchu gradientu (Exploding Gradient Guard).
- **Křivka konvergence (Loss Curve)**: Graf vývoje MSE v čase vs. analytické minimum OLS.
- **Přednastavené datasety a editace**: Výchozí data, vliv odlehlé hodnoty (outlier), silná závislost, editovatelná tabulka a import CSV.
