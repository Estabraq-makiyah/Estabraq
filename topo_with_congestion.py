#!/usr/bin/env python3
"""
Mininet + Faucet topology with:
 • random fault injection (link/trunk/port/Faucet reload),
 • continuous ping floods,
 • periodic UDP congestion floods,
 • structured fault logging (START/END/SUMMARY with duration).
"""
import os
import subprocess
import random
import time
import threading
import signal
import atexit
from datetime import datetime

from mininet.topo  import Topo
from mininet.net   import Mininet
from mininet.node  import RemoteController
from mininet.link  import TCLink
from mininet.cli   import CLI
from mininet.log   import setLogLevel, info

# ───── Configuration ────────────────────────────────────────
FAUCET_PORT     = 6653
# Assumes your faucet config is at:
FAUCET_CONFIG   = "/etc/faucet/faucet.yaml"
FAUCET_LOG_DIR  = "/tmp/faucet"
FAULT_LOG       = "faults.log"

# Random‐fault timing (s)
RND_INTERVAL    = (10, 90)   # between faults
RND_DURATION    = (10, 40)   # how long each fault lasts

# Congestion flood timing & rates
CONG_INTERVAL   = (30, 90)   # between congestion events
CONG_DURATION   = (10, 20)   # UDP flood duration
CONG_BW         = (50, 150)  # Mbps

# ───── Topology ──────────────────────────────────────────────
class RedundantTopo(Topo):
    def build(self):
        # 3‐switch triangle with one office host (h1), one office host (h2) and one guest host (g1)
        s1 = self.addSwitch("sw1")
        s2 = self.addSwitch("sw2")
        s3 = self.addSwitch("sw3")

        h1 = self.addHost("h1", ip="10.0.100.1/24")
        h2 = self.addHost("h2", ip="10.0.100.2/24")
        g1 = self.addHost("g1", ip="10.0.200.1/24")

        # Access links
        self.addLink(h1, s1, port2=1)
        self.addLink(h2, s2, port2=1)
        self.addLink(g1, s3, port2=1)

        # Redundant trunks
        self.addLink(s1, s2, port1=5,  port2=24)
        self.addLink(s1, s3, port1=6,  port2=23)
        self.addLink(s2, s3, port1=25, port2=24)

# ───── Helpers ──────────────────────────────────────────────
def cleanup():
    """Kill Mininet & OVS leftovers."""
    info("*** Cleaning up Mininet & OVS\n")
    os.system("mn -c")
    for br in subprocess.getoutput("ovs-vsctl list-br").splitlines():
        os.system(f"ovs-vsctl del-br {br}")

def start_faucet():
    """Launch Faucet under Ryu (assumes default ofp port 6653 & /etc/faucet/faucet.yaml)."""
    os.makedirs(FAUCET_LOG_DIR, exist_ok=True)
    proc = subprocess.Popen(["ryu-manager", "faucet.faucet", "--use-stderr"])
    atexit.register(proc.terminate)
    info(f"*** Faucet PID {proc.pid}\n")
    return proc

def log_fault(tag, msg):
    """Write a structured fault‐event line."""
    ts = datetime.utcnow().isoformat()
    line = f"{ts} {tag} {msg}"
    print("***", line)
    with open(FAULT_LOG, "a") as f:
        f.write(line + "\n")

# ───── Fault injector ───────────────────────────────────────
def fault_injector(net, faucet_proc):
    switches = {sw.name: sw for sw in net.switches}
    links    = net.links

    while True:
        time.sleep(random.randint(*RND_INTERVAL))
        kind     = random.choice(["link_down","trunk_down","port_toggle","restart_faucet"])
        start_ts = datetime.utcnow()

        if kind == "link_down":
            h = random.choice(net.hosts)
            intf = h.defaultIntf().name
            log_fault("START", f"LINK_DOWN {intf}")
            h.defaultIntf().ifconfig("down")
            time.sleep(random.randint(*RND_DURATION))
            h.defaultIntf().ifconfig("up")
            log_fault("END", f"LINK_UP {intf}")

        elif kind == "trunk_down":
            l = random.choice([l for l in links if "sw" in l.intf1.name])
            n1,n2 = l.intf1.name, l.intf2.name
            log_fault("START", f"TRUNK_DOWN {n1}<->{n2}")
            l.intf1.ifconfig("down"); l.intf2.ifconfig("down")
            time.sleep(random.randint(*RND_DURATION))
            l.intf1.ifconfig("up");   l.intf2.ifconfig("up")
            log_fault("END",   f"TRUNK_UP {n1}<->{n2}")

        elif kind == "port_toggle":
            sw = random.choice(list(switches.values()))
            p  = random.choice(range(1,7))
            log_fault("START", f"PORT_DOWN {sw.name}:{p}")
            subprocess.call(["ovs-ofctl","mod-port",sw.name,str(p),"down"])
            time.sleep(random.randint(*RND_DURATION))
            subprocess.call(["ovs-ofctl","mod-port",sw.name,str(p),"up"])
            log_fault("END",   f"PORT_UP {sw.name}:{p}")

        else:  # restart_faucet
            log_fault("START", "FAUCET_RELOAD")
            faucet_proc.send_signal(signal.SIGHUP)
            log_fault("END",   "FAUCET_RELOAD_DONE")

        # always write a summary with duration
        dur = (datetime.utcnow() - start_ts).total_seconds()
        summary = {
            "link_down":"LINK_FAULT",
            "trunk_down":"TRUNK_FAULT",
            "port_toggle":"PORT_FAULT",
            "restart_faucet":"FAUCET_RELOAD"
        }[kind]
        log_fault("SUMMARY", f"{summary} duration={dur:.1f}s")

# ───── Ping floods ─────────────────────────────────────────
def start_ping_floods(net):
    """Keep pinging between all pairs every 0.2 s."""
    import itertools
    for a,b in itertools.combinations(net.hosts,2):
        a.cmd(f"ping -i 0.2 {b.IP()} &")

# ───── Congestion floods ────────────────────────────────────
def congestion_injector(net):
    """Fire a random UDP flood (iperf3) between h1→g1 to simulate congestion."""
    hA, hB = net.get("h1"), net.get("g1")
    while True:
        time.sleep(random.randint(*CONG_INTERVAL))
        dur = random.randint(*CONG_DURATION)
        bw  = random.randint(*CONG_BW)
        info(f"*** CONGESTION: {hA.name}->{hB.name} @ {bw}Mbps for {dur}s\n")
        hA.cmd(f"iperf3 -u -b {bw}M -t {dur} -c {hB.IP()} &")

# ───── Main ─────────────────────────────────────────────────
def main():
    setLogLevel("info")
    cleanup()

    topo = RedundantTopo()
    net  = Mininet(topo=topo,
                   link=TCLink,
                   controller=lambda name: RemoteController(name,
                                                            ip="127.0.0.1",
                                                            port=FAUCET_PORT),
                   autoSetMacs=True)
    net.start()

    faucet_proc = start_faucet()

    # background threads
    threading.Thread(target=fault_injector,    args=(net, faucet_proc), daemon=True).start()
    threading.Thread(target=start_ping_floods,  args=(net,),            daemon=True).start()
    threading.Thread(target=congestion_injector,args=(net,),            daemon=True).start()

    info("*** Network up — use Mininet CLI (Ctrl‑D to quit)\n")
    CLI(net)
    net.stop()

if __name__ == "__main__":
    main()
