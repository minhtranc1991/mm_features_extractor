import os
import csv
import time
import json
import datetime
from redis import Redis

# Redis connection
r = Redis(host='localhost', port=6379, decode_responses=True)

def extract_lob_features(r, base: str, ex: str, depth_n=5):
    """
    Extract structured LOB features for ML volatility training.
    Returns:
        ts_ms, ts_iso, best_bid, best_ask, last_price,
        best_bid_qty, best_ask_qty,
        bid_notional_n, ask_notional_n
    """
    # Load ticker
    key_ticker = f"{base}_USDT_{ex}_ticker"
    raw_ticker = r.get(key_ticker)
    if raw_ticker is None:
        return None
    tk = json.loads(raw_ticker)
    last_price = float(tk.get("lastPr")) if tk.get("lastPr") else None

    # Load orderbook
    key_ob = f"{base}_USDT_{ex}_orderbook"
    raw_ob = r.get(key_ob)
    if raw_ob is None:
        return None
    ob = json.loads(raw_ob)
    ts_ms = int(ob["ts"])
    ts_iso = datetime.datetime.utcfromtimestamp(ts_ms / 1000).isoformat() + "Z"

    # Parse and sort bids/asks
    bids = sorted([(float(p), float(s)) for p, s in ob["bids"].items()], key=lambda x: -x[0])
    asks = sorted([(float(p), float(s)) for p, s in ob["asks"].items()], key=lambda x: x[0])

    if not bids or not asks:
        return None

    best_bid, best_bid_qty = bids[0]
    best_ask, best_ask_qty = asks[0]

    bid_notional_n = sum(price * size for price, size in bids[:depth_n])
    ask_notional_n = sum(price * size for price, size in asks[:depth_n])

    return [
        ts_ms,
        ts_iso,
        best_bid,
        best_ask,
        last_price,
        best_bid_qty,
        best_ask_qty,
        bid_notional_n,
        ask_notional_n,
    ]

def get_daily_filename(base, ex, base_dir="/var/www/MarketMakerMinh/strategies/mm_bitget/bitget_LOB"):
    today = datetime.datetime.utcnow().strftime("%Y%m%d")
    filename = f"{base.lower()}_{ex.lower()}_lob5_{today}.csv"
    return os.path.join(base_dir, filename)

def start_lob_collector(r, base="CAKE", ex="binance", base_dir="/var/www/MarketMakerMinh/strategies/mm_bitget/bitget_LOB"):
    header = [
        "ts_ms", "ts_iso", "best_bid", "best_ask", "last_price",
        "best_bid_qty", "best_ask_qty", "bid_notional_5", "ask_notional_5"
    ]

    # Ensure output directory exists
    os.makedirs(base_dir, exist_ok=True)

    current_file_path = None
    csvfile = None
    writer = None

    print(f"📡 LOB collector started for {base}/{ex} → writing to daily files in {base_dir}")

    try:
        while True:
            row = extract_lob_features(r, base, ex)
            if not row:
                time.sleep(0.9)
                continue

            ts_ms = row[0]
            current_day_file = get_daily_filename(base, ex, base_dir)

            # Rotate file if day changes
            if current_file_path != current_day_file:
                if csvfile:
                    csvfile.close()
                current_file_path = current_day_file

                write_header = not os.path.exists(current_file_path)
                csvfile = open(current_file_path, "a", newline="", buffering=1)  # line buffering
                writer = csv.writer(csvfile)
                if write_header:
                    writer.writerow(header)

            # Write row
            writer.writerow(row)
            time.sleep(0.9)

    except KeyboardInterrupt:
        print("\n🛑 Collector stopped by user.")
    finally:
        if csvfile:
            csvfile.close()

# ----- RUN -----
if __name__ == "__main__":
    start_lob_collector(r, base="CAKE", ex="binance")