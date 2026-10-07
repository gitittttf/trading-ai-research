import datetime as dt
import hashlib
import io
import os
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import fetch_binance_vision as fbv  # noqa: E402


def _zip(name: str, text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, text)
    return buf.getvalue()


KLINES_HEADER = ("open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,"
                 "taker_buy_quote_volume,ignore\n")
ROW1 = "1704067200000,10.0,10.5,9.5,10.2,100.0,1704067259999,1000.0,10,60.0,600.0,0\n"
ROW2 = "1704067260000,10.2,10.3,10.1,10.25,50.0,1704067319999,500.0,5,20.0,200.0,0\n"


def test_parse_klines_with_and_without_header_and_microseconds():
    a = fbv.parse_klines_csv((KLINES_HEADER + ROW1 + ROW2).encode())
    b = fbv.parse_klines_csv((ROW1 + ROW2).encode())
    micro = ROW1.replace("1704067200000", "1704067200000000", 1)
    c = fbv.parse_klines_csv(micro.encode())
    assert a.columns == ["timestamp", "open", "high", "low", "close", "volume", "taker_buy_volume"]
    assert a.to_dicts() == b.to_dicts()
    assert a["timestamp"].to_list() == [1704067200000, 1704067260000]
    assert a["taker_buy_volume"].to_list() == [60.0, 20.0]
    assert c["timestamp"].to_list() == [1704067200000]


def test_parse_klines_integer_looking_volumes_then_decimals():
    rows = "".join(f"{1704067200000 + i * 60000},10,11,9,10,{100 + i},0,0,1,50,0,0\n" for i in range(200))
    rows += "1704079200000,10,11,9,10,6701.80,0,0,1,50.5,0,0\n"
    df = fbv.parse_klines_csv((KLINES_HEADER + rows).encode())
    assert df["volume"].to_list()[-1] == 6701.80 and df.height == 201


def test_parse_funding():
    raw = "calc_time,funding_interval_hours,last_funding_rate\n1704067200000,8,0.00010000\n1704096000000,8,-0.00005\n"
    df = fbv.parse_funding_csv(raw.encode())
    assert df["timestamp"].to_list() == [1704067200000, 1704096000000]
    assert df["funding_rate"].to_list() == [0.0001, -0.00005]


def test_checksum_verified_and_mismatch_rejected(tmp_path):
    blob = _zip("X-1m-2024-01.csv", KLINES_HEADER + ROW1)
    good = f"{hashlib.sha256(blob).hexdigest()}  X-1m-2024-01.zip\n".encode()
    bad = f"{'0' * 64}  X-1m-2024-01.zip\n".encode()
    store = {"u": blob, "u.CHECKSUM": good}
    out = fbv.fetch_verified("u", str(tmp_path / "a.zip"), fetch=lambda url: store.get(url))
    assert out == blob and (tmp_path / "a.zip").exists()
    store["u.CHECKSUM"] = bad
    with pytest.raises(fbv.ChecksumError):
        fbv.fetch_verified("u", str(tmp_path / "b.zip"), fetch=lambda url: store.get(url))
    assert not (tmp_path / "b.zip").exists()
    # missing file (404) -> None, not an error
    assert fbv.fetch_verified("missing", str(tmp_path / "c.zip"), fetch=lambda url: None) is None


def test_checksum_cache_handles_non_ascii_symbol_names(tmp_path):
    # the archive has perps like 币安人生USDT; their checksum text names the file. The cache must
    # not depend on the platform's default encoding (cp1252 on Windows cannot encode these)
    blob = _zip("币安人生USDT-1d-2025-10.csv", KLINES_HEADER + ROW1)
    chk = f"{hashlib.sha256(blob).hexdigest()}  币安人生USDT-1d-2025-10.zip\n".encode()
    store = {"u": blob, "u.CHECKSUM": chk}
    path = str(tmp_path / "币安人生USDT-1d-2025-10.zip")
    assert fbv.fetch_verified("u", path, fetch=lambda url: store.get(url)) == blob
    # second call reads the cached file and its checksum back
    assert fbv.fetch_verified("u", path, fetch=lambda url: None) == blob


def test_url_plan_monthly_then_daily():
    urls = fbv.klines_urls("DOTUSDT", "2026-08", dt.date(2026, 10, 3))
    names = [u.rsplit("/", 1)[1] for u, _ in urls]
    assert names == ["DOTUSDT-1m-2026-08.zip", "DOTUSDT-1m-2026-09.zip",
                     "DOTUSDT-1m-2026-10-01.zip", "DOTUSDT-1m-2026-10-02.zip"]
    assert fbv.exchange_symbol("1000PEPE/USDT") == "1000PEPEUSDT"


def test_build_symbol_end_to_end_with_fake_server(tmp_path, monkeypatch):
    monkeypatch.setattr(fbv, "RAW_DIR", str(tmp_path / "raw"))
    k = _zip("DOTUSDT-1m-2024-01.csv", KLINES_HEADER + ROW1 + ROW2)
    f = _zip("DOTUSDT-fundingRate-2024-01.csv",
             "calc_time,funding_interval_hours,last_funding_rate\n1704067200000,8,0.0001\n")
    store = {}
    for name, blob, kind in [("DOTUSDT-1m-2024-01.zip", k, "klines/DOTUSDT/1m"),
                             ("DOTUSDT-fundingRate-2024-01.zip", f, "fundingRate/DOTUSDT")]:
        url = f"{fbv.BASE_URL}/monthly/{kind}/{name}"
        store[url] = blob
        store[url + ".CHECKSUM"] = f"{hashlib.sha256(blob).hexdigest()}  {name}".encode()
    rep = fbv.build_symbol("DOT/USDT", "2024-01", dt.date(2024, 2, 1), str(tmp_path), 2,
                           fetch=lambda url: store.get(url))
    assert rep["kline_rows"] == 2 and rep["funding_rows"] == 1
    from core.data import load_funding, load_klines
    assert load_klines("DOT/USDT", str(tmp_path))["close"].to_list() == [10.2, 10.25]
    assert load_funding("DOT/USDT", str(tmp_path))["funding_rate"].to_list() == [0.0001]


def test_missing_recent_month_is_filled_from_daily_files(tmp_path, monkeypatch):
    monkeypatch.setattr(fbv, "RAW_DIR", str(tmp_path / "raw"))
    store = {}

    def put(url, name, rows):
        blob = _zip(name.replace(".zip", ".csv"), KLINES_HEADER + rows)
        store[url] = blob
        store[url + ".CHECKSUM"] = f"{hashlib.sha256(blob).hexdigest()}  {name}".encode()

    # January monthly exists, February monthly NOT published yet -> daily files must be used
    put(f"{fbv.BASE_URL}/monthly/klines/DOTUSDT/1m/DOTUSDT-1m-2024-01.zip", "DOTUSDT-1m-2024-01.zip", ROW1)
    feb1 = "1706745600000,10.0,10.5,9.5,10.2,100.0,1706745659999,1000.0,10,60.0,600.0,0\n"
    put(f"{fbv.BASE_URL}/daily/klines/DOTUSDT/1m/DOTUSDT-1m-2024-02-01.zip", "DOTUSDT-1m-2024-02-01.zip", feb1)
    rep = fbv.build_symbol("DOT/USDT", "2024-01", dt.date(2024, 3, 2), str(tmp_path), 2,
                           fetch=lambda url: store.get(url), max_gap_days=40)
    assert rep["daily_fallback_files"] == 1 and rep["kline_rows"] == 2
    with pytest.raises(SystemExit):
        fbv.build_symbol("DOT/USDT", "2024-01", dt.date(2024, 3, 2), str(tmp_path), 2,
                         fetch=lambda url: store.get(url), max_gap_days=3)
