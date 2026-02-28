 #!/usr/bin/env python3
import os
import sys
import time
import csv
import requests
from datetime import datetime

# ---- Configuration ----
PROMETHEUS_URL = "http://localhost:9090"
METRICS_PREFIX = "faucet_"
SCRAPE_INTERVAL = 5  # seconds

FAULT_LOG      = "faults.log"
OUTPUT_CSV     = "rl_features.csv"
FIELDNAMES = [
    "timestamp",
    # process metrics
    "process_cpu_seconds_total",
    "process_resident_memory_bytes",
    "process_virtual_memory_bytes",
    # OF stats (via Prometheus)
    "of_flowmsgs_sent_total",
    "of_errors_total",
    "of_dp_connections_total",
    "of_dp_disconnections_total",
    # port_status aggregated (count down ports)
    "ports_down_count",
    # fault annotation (from last fault summary)
    "last_fault"
]

def ensure_fault_log():
    if not os.path.isfile(FAULT_LOG):
        open(FAULT_LOG, "w").close()
        print(f"[DEBUG] Created empty {FAULT_LOG}", flush=True)

def load_last_fault():
    """Read the last SUMMARY line from faults.log, or ''."""
    last = ""
    with open(FAULT_LOG) as f:
        for line in f:
            parts = line.strip().split(" ", 2)
            if len(parts) == 3 and parts[1] == "SUMMARY":
                last = parts[2]
    return last

def query_metric(name):
    """Instant query of a single metric (returns float or 0)."""
    r = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": name})
    r.raise_for_status()
    data = r.json().get("data", {}).get("result", [])
    if data and len(data[0].get("value", [])) == 2:
        return float(data[0]["value"][1])
    return 0.0

def count_ports_down():
    """Count how many ports have port_status == 0."""
    r = requests.get(f"{PROMETHEUS_URL}/api/v1/query",
                     params={"query": 'port_status==0'})
    r.raise_for_status()
    data = r.json().get("data", {}).get("result", [])
    return len(data)

def init_csv():
    write_header = not os.path.isfile(OUTPUT_CSV)
    f = open(OUTPUT_CSV, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()
    return f, writer

def main():
    print("[INFO] Starting RL feature collector", flush=True)
    ensure_fault_log()
    csv_file, writer = init_csv()

    try:
        while True:
            ts = datetime.utcnow().isoformat()
            row = {"timestamp": ts}

            # Core process metrics
            row["process_cpu_seconds_total"]     = query_metric("process_cpu_seconds_total")
            row["process_resident_memory_bytes"] = query_metric("process_resident_memory_bytes")
            row["process_virtual_memory_bytes"]  = query_metric("process_virtual_memory_bytes")

            # OF-related metrics
            row["of_flowmsgs_sent_total"]       = query_metric("of_flowmsgs_sent_total")
            row["of_errors_total"]              = query_metric("of_errors_total")
            row["of_dp_connections_total"]      = query_metric("of_dp_connections_total")
            row["of_dp_disconnections_total"]   = query_metric("of_dp_disconnections_total")

            # Port down count
            row["ports_down_count"] = count_ports_down()

            # Last fault summary
            row["last_fault"] = load_last_fault()

            writer.writerow(row)
            csv_file.flush()

            print(f"[SCRAPE {ts}] wrote row, last_fault='{row['last_fault']}'", flush=True)
            time.sleep(SCRAPE_INTERVAL)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted, exiting.", flush=True)
    finally:
        csv_file.close()

if __name__ == "__main__":
    # Force unbuffered stdout
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    main()
