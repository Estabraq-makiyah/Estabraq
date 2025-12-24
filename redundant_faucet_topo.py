#!/usr/bin/env python3
"""
Mininet + Faucet topology with random fault injection (no Gauge).

Faults are logged to faults.log with START, END, and SUMMARY (with duration).
"""

import os
import subprocess
import random
import time
import threading
import signal
import atexit
from datetime import datetime

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI

# ───── Global configuration ───────────────────────────────────────────────
FAUCET_PORT   = 6653
FAUCET_CONFIG = "/etc/faucet/faucet.yaml"
FAULT_LOG     = "faults.log"

# Random intervals/durations for faults (secs)
RND_INTERVAL  = (10, 60)
RND_DURATION  = (10, 30)

# ───── Topology definition ────────────────────────────────────────────────
class RedundantTopo(Topo):
    def build(self):
        # Three switches in a triangle, each with one host
        s1 = self.addSwitch('sw1')
        s2 = self.addSwitch('sw2')
        s3 = self.addSwitch('sw3')

        h1 = self.addHost('h1', ip='10.0.100.1/24')
        h2 = self.addHost('h2', ip='10.0.100.2/24')
        g1 = self.addHost('g1', ip='10.0.200.1/24')

        # Access links
        self.addLink(h1, s1, port2=1)
        self.addLink(h2, s2, port2=1)
        self.addLink(g1, s3, port2=1)

        # Redundant trunks
        self.addLink(s1, s2, port1=5, port2=24)
        self.addLink(s1, s3, port1=6, port2=23)
        self.addLink(s2, s3, port1=25, port2=24)

# ───── Cleanup stale Mininet/OVS artifacts ────────────────────────────────
def cleanup():
    info("*** Removing old Mininet/OVS state\n")
    os.system("mn -c")
    for br in subprocess.getoutput("ovs-vsctl list-br").splitlines():
        os.system(f"ovs-vsctl del-br {br}")

# ───── Launch Faucet under Ryu ─────────────────────────────────────────────
def start_faucet() -> subprocess.Popen:
    info("*** Starting Faucet\n")
    cmd = [
        "ryu-manager", "faucet.faucet",
        "--ryu-ofp-tcp-listen-port", str(FAUCET_PORT),
        "--ryu-config-file",       FAUCET_CONFIG,
        "--use-stderr"
    ]
    proc = subprocess.Popen(cmd)
    atexit.register(proc.terminate)
    info(f"*** Faucet PID {proc.pid}\n")
    return proc

# ───── Structured fault logging ────────────────────────────────────────────
def log_fault(tag: str, msg: str):
    ts = datetime.utcnow().isoformat()
    line = f"{ts} {tag} {msg}"
    print("***", line)
    with open(FAULT_LOG, "a") as fh:
        fh.write(line + "\n")

# ───── Fault‑injection thread ──────────────────────────────────────────────
def fault_injector(net: Mininet, faucet_proc: subprocess.Popen):
    switches = { sw.name: sw for sw in net.switches }
    links    = net.links

    while True:
        # wait a random time before next fault
        time.sleep(random.randint(*RND_INTERVAL))

        kind     = random.choice(["link_down","trunk_down","port_toggle","restart_faucet"])
        start_ts = datetime.utcnow()
        start_if = start_ts.isoformat()

        if kind == "link_down":
            host = random.choice(net.hosts)
            intf = host.defaultIntf()
            log_fault("START", f"LINK_DOWN {intf}")
            intf.ifconfig("down")
            time.sleep(random.randint(*RND_DURATION))
            intf.ifconfig("up")
            dur = (datetime.utcnow() - start_ts).total_seconds()
            log_fault("END", f"LINK_UP   {intf}")
            log_fault("SUMMARY", f"LINK_FAULT {intf} duration={dur:.1f}s")

        elif kind == "trunk_down":
            trunk = random.choice([l for l in links 
                                   if l.intf1.node.name.startswith("sw") 
                                   and l.intf2.node.name.startswith("sw")])
            a, b = trunk.intf1, trunk.intf2
            log_fault("START", f"TRUNK_DOWN {a.name}<->{b.name}")
            a.ifconfig("down"); b.ifconfig("down")
            time.sleep(random.randint(*RND_DURATION))
            a.ifconfig("up");   b.ifconfig("up")
            dur = (datetime.utcnow() - start_ts).total_seconds()
            log_fault("END",   f"TRUNK_UP   {a.name}<->{b.name}")
            log_fault("SUMMARY", f"TRUNK_FAULT {a.name}<->{b.name} duration={dur:.1f}s")

        elif kind == "port_toggle":
            sw   = random.choice(list(switches.values()))
            port = random.choice(range(1,7))
            log_fault("START", f"PORT_DOWN {sw.name}:{port}")
            subprocess.call(["ovs-ofctl","mod-port",sw.name,str(port),"down"])
            time.sleep(random.randint(*RND_DURATION))
            subprocess.call(["ovs-ofctl","mod-port",sw.name,str(port),"up"])
            dur = (datetime.utcnow() - start_ts).total_seconds()
            log_fault("END",   f"PORT_UP   {sw.name}:{port}")
            log_fault("SUMMARY", f"PORT_FAULT {sw.name}:{port} duration={dur:.1f}s")

        else:  # restart_faucet
            log_fault("START", "FAUCET_RELOAD")
            faucet_proc.send_signal(signal.SIGHUP)
            log_fault("END",   "FAUCET_RELOAD_DONE")
            log_fault("SUMMARY", "FAUCET_RELOAD duration=0s")

# ───── Main entry ─────────────────────────────────────────────────────────
def main():
    setLogLevel("info")
    cleanup()

    topo = RedundantTopo()
    net  = Mininet(
        topo=topo, link=TCLink,
        controller=lambda name: RemoteController(name, ip="127.0.0.1", port=FAUCET_PORT),
        autoSetMacs=True
    )
    net.start()

    faucet_proc = start_faucet()
    threading.Thread(target=fault_injector, args=(net, faucet_proc), daemon=True).start()

    info("*** Network running – use CLI or Ctrl‑D to stop\n")
    CLI(net)
    net.stop()

if __name__ == "__main__":
    main()
