# AGENTS.md

給在這個 repo 工作的 AI agent 的說明。**這是公開 repo**,寫進來的東西所有人看得到。

`README.md` 講「這個專案在做什麼、結論是什麼」。這份講「怎麼動它、哪裡會踩到雷」。

## 一句話

台股 + 美股的動能掃描:每天抓全市場收盤、算篩選清單與市場廣度、用新聞判斷族群題材,
輸出成 `docs/` 底下的靜態網頁(GitHub Pages)。另外有台指期選擇權的 GEX 位階,
以及回測與統計驗證的腳本。

## 資料流

```
交易所 API ─┐
            ├─ scripts/update_data.py  ─→ data/history.csv.gz(滾動 280 個交易日)
            │                             data/index.csv.gz  (加權指數, RS 的基準)
            │                             data/breadth.csv   (每日通過篩選的檔數)
            │                             data/momentum_members.csv.gz(每日名單+題材)
            │                             docs/data/latest.json ← 網頁讀這個
            │      └─ scripts/themes.py     題材判斷(Google News 關鍵字 + 覆寫檔)
            │
Nasdaq/Yahoo ─ scripts/us_breadth.py   ─→ data/us_breadth.csv / us_momentum_members.csv.gz
            │      └─ scripts/us_themes.py  美股題材(與 themes.py 共用引擎)
            │                             docs/data/us_latest.json
            │
期交所 ─────── scripts/txo_gex.py       ─→ data/gex_history.csv / docs/data/gex_latest.json
```

`.github/workflows/daily.yml` 每天跑前三支並 commit 結果。前端 `docs/index.html`
是單檔無框架的靜態頁,直接 fetch 那三個 JSON。

## 常用指令

```bash
python scripts/update_data.py                 # 每日更新(補上區間內缺的交易日)
python scripts/update_data.py --recompute     # 不抓新資料, 用現有歷史重算 latest.json
python scripts/update_data.py --rebuild-breadth  # 改了篩選參數後, 整段重算廣度與名單
python scripts/update_data.py --backfill      # 首次建立歷史

python scripts/us_breadth.py                  # 美股每日更新(抓一年日K)
python scripts/us_breadth.py --backfill       # 抓兩年, 重算約 280 天廣度與名單

python scripts/txo_gex.py                     # GEX 每日更新
python scripts/backtest_momentum.py --start 2023-11-06 --r 0.07,0.15   # 回測
python scripts/breadth_study.py --threshold 100                        # 廣度事件研究
```

## 改東西之前必讀

### 改篩選條件 → 一定要整段重算

篩選條件(`update_data.py` 頂端的 `BR_*`)一改,**歷史廣度就跟新定義對不上**。
不重算的話,`breadth.csv` 前面是舊定義、之後是新定義,那條線會在改的當天出現一個
不存在的斷層,20MA、上榜天數、族群日增減也全部跟著錯。

台股跑 `--rebuild-breadth`,美股跑 `--backfill`。兩者都會保留既有的題材標籤。

回測與研究的名單快取檔名帶著 `SCREEN_SIG`(條件指紋),條件一改自動失效,
不會悄悄拿舊定義的名單去算新問題。

### 終端機是 cp950,檔案是 UTF-8

直接 `print` 中文會亂碼,`python -c "..."` 裡放中文也會。要看中文結果就**寫到暫存檔再用
Read 工具讀**。這不是顯示問題而已 —— 用 `-c` 傳中文字串進去會被改掉,測試結果會是錯的。

多層跳脫(heredoc → Python 字串 → 寫進檔案)容易把 `\b`、`\n` 弄成退格字元或真正的換行。
寫含正則或跳脫序列的程式碼,用 Edit/Write 工具,不要用 heredoc 疊字串。

### 資料來源的雷

* **價格未還原權值**。交易所的每日行情 API 給原始價,除權息/減資/面額變更當天會出現
  超過 ±10% 的「跌幅」(2026 年就有 66 次)。台股有漲跌幅限制,所以單日變動超過 10.5%
  一定是公司行為,不是行情。回測用這個特徵偵測並排除,不要當成虧損。
* **欄位名稱用比對找,不要寫死**。發行股數那欄實際叫「已發行普通股數或TDR原股發行股數」,
  少抄一個字會安靜地全部對不上、不報錯。
* **`data/index.csv.gz` 一定要進 commit**。它是 RS 的基準(63/126/189/252 日報酬),
  workflow 的 `git add` 漏掉過一次,結果大盤歷史一天天缺漏、RS 跟著偏移。
* **TPEx 的 TLS 憑證鏈不完整**。伺服器只送葉憑證,瀏覽器會自己補、Python 不會。
  `scripts/certs/` 放了缺的那張中繼憑證,驗證仍是完整一條鏈到受信任的根,不是關掉驗證。
* **Yahoo 盤中會回一根沒收完的 K 棒**。美股每日更新遇到已有紀錄的日子會跳過,
  所以盤中算出來的半根會永久留著。紐約時間 16:20 前一律丟掉當天那根。
* **JSON 不能有 NaN**。`json.dumps` 預設會輸出裸 `NaN`,Python 讀得回來但瀏覽器的
  `JSON.parse` 會整個失敗、網頁空白。一律 `allow_nan=False`,並逐格把 NaN 轉成 None
  (`DataFrame.where(cond, None)` 對 float 欄位是 no-op,不能用)。

### 排程不可靠

GitHub 的 `schedule` 對公開 repo 是 best-effort。實測 30 個交易日裡從沒跑滿設定的班數,
有一天三班全部跳過。所以:

* 流水線設計成**補洞式**:掃最近幾個日曆日,補上任何還沒進歷史的交易日。重跑沒有副作用。
* 真正的觸發來源是外部 cron 打 `workflow_dispatch`,GitHub 自己的 cron 只當備援。

### 公開 repo,有些東西不能進 git

* `refs/` 是使用者自己的參考資料,一律不進 git。
* `scripts/xq_*` 是使用者未完成的工作檔,**不要 commit**,除非他明講。
* 加長歷史、名單快取、根目錄的 `_*.txt` / `_*.csv` 都是可重新產生的產物,已在 `.gitignore`。
* 不要把 token、金鑰寫進任何檔案或 commit 訊息。

## 題材覆寫檔

`data/theme_overrides.csv`(美股 `us_theme_overrides.csv`)優先於關鍵字判斷,
由本機的排程研究任務維護。欄位 `code,theme,source,updated`,
`source` 固定寫「本文讀到的事實 | 本文: 網址」,方便事後查核。

`theme` 填 **`-`** 代表「讀過本文、確定是個股因素或投機,不屬於任何族群」,跟沒寫不同:

* `-` → 網頁標 ✓、研究任務跳過、蓋掉關鍵字誤判
* 沒寫 → 還沒查或讀不到本文,下次再查

算族群日增減時,`-` 必須跟著套用到前一天,否則被蓋掉的個股會退回歷史上那個錯的標籤,
只要它還在榜上就每天憑空多一組差值。

## 做統計驗證時

這個 repo 的結論幾乎都是負面的,而且是靠方法上的謹慎才發現是負面的。沿用同樣的標準:

* **跟合理的基準比,不是跟零比**。多頭期間任何一段的期望報酬都是正的。
  「牆的區間能框住隔日價格」要對照的是「同寬度、以現價為中心的框」—— 實測 GEX 的牆
  比那個還差 8~10 個百分點。
* **事件會連在一起**。廣度在門檻附近會來回穿越,不做不應期會把一段算成好幾個獨立樣本,
  顯著性整個灌水(實測 28 次 → 去重後 16 次)。
* **一口氣測很多條件就會生出假發現**。測了六個「第二層濾網」都無效,其中一個方向相反
  且樣本小 —— 那是雜訊,不是訊號,不要拿去用。
* **參數結論要跨期間驗證**。「停損放寬比較好」在 2026 成立,拉到 2023-11 起的近三年
  仍成立但獲利大幅縮水,而且 2025 每個停損寬度都虧。單一年度的結論不算數。

## 目前已知的結論(不要重新發明)

* **GEX 的位階沒有可用的預測價值**,詳見 README。已經用公平的對照組驗證過兩次。
* **動能篩選當第一層可以,當完整策略不行**:三年回測沒有一年打敗大盤。
  最大的兩個槓桿是停損寬度和交易成本,都跟選股無關。
* **關鍵字在美股幾乎不管用**(RSS 被內容農場洗版),所以美股靠人工/LLM 研究,
  關鍵字只是補位;台股相反。
