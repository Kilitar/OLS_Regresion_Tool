# 📐 Interaktivní vizualizátor Metody nejmenších čtverců (OLS)

Tato aplikace umožňuje interaktivně zkoumat princip **Metody nejmenších čtverců (Ordinary Least Squares - OLS)** v lineární regresi. Uživatel může posouvat a naklápět regresní přímku a v reálném čase sledovat, jak se mění velikost vertikálních odchylek (reziduí) a geometrických čtverců chyb ($e_i^2$).

---

## 🚀 Jak aplikaci spustit lokálně

### 1. Varianta: Streamlit (Doporučeno)

1. Ujistěte se, že máte nainstalovaný Python 3.9+ a knihovny:
   ```bash
   pip install -r requirements.txt
   ```
2. Spusťte aplikaci:
   ```bash
   streamlit run app.py
   ```
3. Aplikace se automaticky otevře ve vašem prohlížeči na adrese `http://localhost:8501`.

### 2. Varianta: Samostatná webová verze (Zero-install)
Otevřete přímo soubor `index.html` v libovolném webovém prohlížeči (dvouklik na soubor). Nepotřebuje žádný server ani Python.

---

## 🌐 Jak aplikaci nasadit ZDARMA ONLINE

### Možnost A: Streamlit Community Cloud (Doporučeno pro Streamlit)
1. Nahrajte tento adresář do nového veřejného repozitáře na **GitHubu** (např. `ols-regression-app`).
2. Přejděte na [share.streamlit.io](https://share.streamlit.io) a přihlaste se přes váš GitHub účet.
3. Klikněte na **"New app"**:
   - **Repository:** vyberte váš repozitář
   - **Branch:** `main` (nebo `master`)
   - **Main file path:** `app.py`
4. Klikněte na **"Deploy!"**.
5. Během cca 1 minuty získáte veřejnou URL (např. `https://ols-regression.streamlit.app`), kterou můžete sdílet komukoliv na internetu!

### Možnost B: GitHub Pages (Pro HTML verzi)
1. V nastavení vašeho GitHub repozitáře přejděte na záložku **Settings -> Pages**.
2. Vyberte větev `main` a složku `/ (root)`.
3. Během okamžiku máte online veřejnou webovou stránku s `index.html`.

---

## 📊 Funkce aplikace

- **Geometrické čtverce reziduí ($e_i^2$)**: Vizuální ukázka toho, proč se metoda nazývá metoda nejmenších *čtverců*.
- **Posuvníky směrnice ($\beta_1$) a posunu ($\beta_0$)**: Možnost manuálního hledání optimální přímky.
- **Tlačítko "Nastavit optimální OLS přímku"**: Okamžitě dosadí analyticky spočtené minimum.
- **Živé metriky**:
  - Aktuální SSE (Sum of Squared Errors)
  - Minimální možné SSE
  - Rozdíl od optima ($\Delta\text{SSE}$)
  - Koeficient determinace $R^2$
- **Různé scénáře dat**: Výchozí data, vliv odlehlé hodnoty (outlier), silná korelace, editovatelná tabulka bodů i možnost nahrát vlastní CSV.
- **Matematické odvození**: Přehledné vysvětlení normálních rovnic a vzorců v LaTeXu.
