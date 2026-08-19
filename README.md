# 台股強勢股掃描

每日盤後自動掃描全市場(上市+上櫃普通股),產生兩份清單:

- **SEPA Trend Template** — Minervini 七條件 + RS Rating(相對大盤超額報酬,全市場百分位 1–99)
- **當日強勢** — 漲幅 ≥ 4%、量比 ≥ 1.5、成交值 ≥ 1 億

同時每日重建台指選擇權(TXO)的 **GEX 位階**(Call Wall / Put Wall / 山頂 / 山谷 / Micro Flip / Macro Zero),座標為期貨遠期價。

架構:GitHub Actions 每個交易日 15:30(台北)自動抓資料、計算、commit 結果 → GitHub Pages 提供靜態前端。零伺服器、零費用。

## 部署步驟

1. 把這個資料夾推上 GitHub(public 或 private repo 皆可,Pages 需 public 或付費方案)
2. **回補歷史資料**(只需做一次):
   - 本機執行:
     ```bash
     pip install pandas requests numpy scipy
     python scripts/update_data.py --backfill
     python scripts/txo_gex.py --backfill
     ```
   - 個股/大盤呼叫證交所、櫃買中心免費歷史 API,約 280 個交易日跑數分鐘;GEX 呼叫期交所選擇權下載,按月切片,通常一兩分鐘內完成。皆無需金鑰、無需註冊
   - commit 產生的 `data/history.csv.gz`、`data/gex_history.csv`、`docs/data/latest.json`、`docs/data/gex_latest.json` 並 push
3. Repo → Settings → Pages → Source 選 `Deploy from a branch`,分支 `main`、目錄 `/docs`
4. Repo → Settings → Actions → General → Workflow permissions 選 **Read and write**
5. 完成。之後每個交易日 15:30 會自動更新,也可到 Actions 頁面手動觸發

## 檔案結構

```
scripts/update_data.py      個股資料管線(抓取、累積歷史、計算、輸出 JSON)
scripts/txo_gex.py          TXO GEX 位階管線(抓取、計算、輸出 JSON)
.github/workflows/daily.yml 每日排程(兩條管線都跑,互不阻擋)
docs/index.html             前端頁面
docs/data/latest.json       個股計算結果(自動產生)
docs/data/gex_latest.json   最新一天 GEX 位階(自動產生)
data/history.csv.gz         滾動 280 個交易日的個股歷史資料(自動維護)
data/index.csv.gz           滾動 280 個交易日的加權股價指數(大盤,自動維護)
data/gex_history.csv        滾動 280 個交易日的 GEX 位階歷史(自動維護)
```

## 篩選邏輯

**SEPA(七條件全過 + RS ≥ 70):**
1. 收盤 > 150MA 且 > 200MA
2. 150MA > 200MA
3. 200MA 上升(對比一個月前)
4. 50MA > 150MA > 200MA
5. 收盤 > 50MA
6. 收盤高於 52 週低點至少 30%
7. 收盤距 52 週高點 25% 以內

RS Rating 採 IBD 式加權:40% 三個月 + 各 20% 六/九/十二個月的**超額報酬**(個股報酬 − 加權股價指數同期報酬),全市場排百分位。上市、上櫃股票統一以加權指數(大盤)為基準,而非各自比較所屬市場的指數。

**當日強勢:** 漲幅 ≥ 4%、量比(當日量/20 日均量)≥ 1.5、成交值 ≥ 1 億。

門檻都在 `scripts/update_data.py` 的 `compute()` 裡,可自行調整。

## GEX 位階

用 TXO(台指選擇權)每日結算價,透過買賣權平價反推期貨遠期價(基差自動內含,不必另抓加權指數估股利率),Black-76 反推 IV 後計算全鏈 dollar gamma,標出六個關鍵價位:

- **Call Wall / Put Wall** — 現價上/下方 net gamma 最大的履約價
- **山頂 Peak** — gross gamma 最大的履約價
- **山谷 Valley** — 現價下方兩個量能區之間的凹陷(跌破後容易加速的真空帶)
- **Micro Flip** — 總 GEX 由負轉正/正轉負的價位(全部到期日)
- **Macro Zero** — 同上,但只用到期日 > 7 天的合約

山谷是找**局部**低點,不是區間最小值 —— gross gamma 隨著遠離價平單調衰減,取全域最小會永遠落在掃描窗最外緣那根空履約價上。實作先把履約價分箱(近月有 50 點檔,量能遠小於百點檔,不分箱會每根都被誤判成谷),再用 prominence 篩深度。找不到夠深的谷就回空值,不硬給數字;指數水位越低、掃描窗內格數越少,越容易出現空值(實測空值率與指數水位相關 -0.88)。

`regime` 為正代表造市商避險行為傾向壓抑價格波動,為負則傾向放大波動。多空 gamma 幾乎抵銷時會標 **neutral** —— 判斷依據是 `net_ratio`(淨 gamma / 總 gamma),低於 `NEUTRAL_RATIO` 就不給正負。這個門檻有實測依據:`net_ratio` ≤ 0.05 的日子,正負號隔日翻面率是 63~65%(比擲銅板還糟),跨過 0.05 掉到 36%,超過 0.2 只剩 12.5%。

結算日當天到期的合約整批排除:期交所對最後交易日的到期序列不發結算價(整批寫 0),反推不出 IV,且收盤後那份 gamma 不再產生盤中避險。

這套方法採靜態假設(造市商 long call / short put),品質過濾門檻(`MIN_OI`、`MIN_PRICE`、`VALLEY_MIN_DEPTH`、`NEUTRAL_RATIO`)會顯著影響結果,都在 `scripts/txo_gex.py` 頂端可調整。

## 資料來源

- 上市個股 + 加權股價指數:證交所 `MI_INDEX`(免金鑰,逐交易日查詢,每日更新與歷史回補共用)
- 上櫃個股:櫃買中心 `dailyQuotes`(免金鑰,逐交易日查詢)
- TXO 選擇權:期交所 `optDataDown`(免金鑰,依期間查詢,單次最長約一個月)

僅供研究參考,不構成投資建議。
