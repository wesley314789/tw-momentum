twca_ssl_subca.pem — TWCA SSL Certification Authority(中繼憑證)

www.tpex.org.tw 在 2026-09-07 換發伺服器憑證後, TLS 交握時只送葉憑證、沒有
附上這張中繼憑證(openssl s_client 量到鏈長 1, Verify return code 21)。瀏覽器
會照葉憑證的 AIA 欄位自己去下載補上, Python/OpenSSL 不會 —— 所以 GitHub
Actions 上會直接 CERTIFICATE_VERIFY_FAILED: unable to get local issuer
certificate, 整條台股 pipeline 中止。

這張就是從葉憑證 AIA 指的位置抓的:
  http://sslserver.twca.com.tw/cacert/Cyber_SSL_2023.crt

subject : C=TW, O=TAIWAN-CA, OU=SSL Sub-CA, CN=TWCA SSL Certification Authority
issuer  : C=TW, O=TAIWAN-CA, OU=Root CA, CN=TWCA CYBER Root CA
有效期  : 2023-02-23 ~ 2033-02-23
SHA-256 : 01:AF:23:24:D0:98:09:8F:5E:0C:DF:6F:AA:BA:DA:43:0B:21:CC:E7:77:F4:7E:AC:B2:62:48:B2:FD:A3:E5:31

它的簽發者 TWCA CYBER Root CA 本來就在 certifi 裡, 所以驗證仍然是完整的一條鏈
到受信任的根, 不是關掉驗證。update_data.py 會把 certifi 的根加上這一張合成一個
暫時的 CA bundle 來用。

TPEx 哪天把伺服器設定修好、或 TWCA 換中繼憑證, 這張就可以拿掉 —— 檢查方式:
  echo | openssl s_client -connect www.tpex.org.tw:443 -servername www.tpex.org.tw
鏈長回到 2 以上且 Verify return code: 0 就不需要了。
