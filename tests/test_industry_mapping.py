"""官方兩市公司基本資料欄位、產業代碼與不完整回應保護。"""

import tempfile
import unittest
from pathlib import Path

from scripts import industry_mapping


class IndustryMappingTest(unittest.TestCase):
    def test_official_fields_and_common_codes(self):
        twse = [{"公司代號": "1101", "產業別": "01"},
                {"公司代號": "2330", "產業別": "24"},
                {"公司代號": "9103", "產業別": "91"},
                {"公司代號": "0050", "產業別": "24"}]
        tpex = [{"SecuritiesCompanyCode": "1240", "SecuritiesIndustryCode": "33"},
                {"SecuritiesCompanyCode": "6488", "SecuritiesIndustryCode": "24"}]
        listed = industry_mapping._rows(twse, market="上市", code_key="公司代號",
                                        industry_key="產業別")
        otc = industry_mapping._rows(tpex, market="上櫃",
                                     code_key="SecuritiesCompanyCode",
                                     industry_key="SecuritiesIndustryCode")
        self.assertEqual({r["code"]: r["sector"] for r in listed},
                         {"1101": "水泥工業", "2330": "半導體業"})
        self.assertEqual({r["code"]: r["sector"] for r in otc},
                         {"1240": "農業科技業", "6488": "半導體業"})

    def test_both_markets_required_then_saved_mapping_loads(self):
        twse = [{"公司代號": str(code), "產業別": "24"}
                for code in range(1000, 1800)]
        tpex = [{"SecuritiesCompanyCode": str(code), "SecuritiesIndustryCode": "28"}
                for code in range(2000, 2700)]
        with self.assertRaises(ValueError):
            industry_mapping.build(twse, tpex[:5])
        result = industry_mapping.build(twse, tpex)
        self.assertEqual(len(result), 1500)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sector_mapping.csv"
            industry_mapping.save(result, path)
            mapping = industry_mapping.load(path)
            self.assertEqual((mapping["1000"], mapping["2000"]),
                             ("半導體業", "電子零組件業"))


if __name__ == "__main__":
    unittest.main()
