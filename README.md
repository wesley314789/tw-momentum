# 台股強勢股掃描

每日盤後自動掃描全市場(上市+上櫃普通股),產生兩份清單:

- **SEPA Trend Template** — Minervini 七條件 + RS Rating(全市場動能百分位 1–99)
- **當日強勢** — 漲幅 ≥ 4%、量比 ≥ 1.5、成交值 ≥ 1 億

架構:GitHub Actions 每個交易日 15:30(台北)自動抓資料、計算、commit 結果 → GitHub Pages 提供靜態前端。零伺服器、零費用。

## 部署步驟

1. 把這個資料夾推上 GitHub(public 或 private repo 皆可,Pages 需 public 或付費方案)
2. **回補歷史資料**(只需做一次):
   - 本機執行:
     ```bash
     pip install pandas requests
     python scripts/update_data.py --backfill
     ```
   - 呼叫證交所/櫃買中心免費歷史 API,每個交易日各 1 次(無需金鑰、無需註冊),約 280 個交易日跑數分鐘
   - commit 產生的 `data/history.csv.gz` 和 `docs/data/latest.json` 並 push
3. Repo → Settings → Pages → Source 選 `Deploy from a branch`,分支 `main`、目錄 `/docs`
4. Repo → Settings → Actions → General → Workflow permissions 選 **Read and write**
5. 完成。之後每個交易日 15:30 會自動更新,也可到 Actions 頁面手動觸發

## 檔案結構

```
scripts/update_data.py      資料管線(抓取、累積歷史、計算、輸出 JSON)
.github/workflows/daily.yml 每日排程
docs/index.html             前端頁面
docs/data/latest.json       計算結果(自動產生)
data/history.csv.gz         滾動 280 個交易日的歷史資料(自動維護)
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

RS Rating 採 IBD 式加權:40% 三個月報酬 + 各 20% 六/九/十二個月報酬,全市場排百分位。

**當日強勢:** 漲幅 ≥ 4%、量比(當日量/20 日均量)≥ 1.5、成交值 ≥ 1 億。

門檻都在 `scripts/update_data.py` 的 `compute()` 裡,可自行調整。

## 資料來源

- 每日更新:證交所 OpenAPI(`STOCK_DAY_ALL`)+ 櫃買中心 OpenAPI(免金鑰)
- 歷史回補:證交所 `MI_INDEX`+ 櫃買中心 `dailyQuotes`(免金鑰,逐交易日查詢)

僅供研究參考,不構成投資建議。
