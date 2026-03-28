#!/usr/bin/env python3
"""Fat-tree k=4: 16 switches, 16 hosts for scalability evaluation."""

import os, subprocess, random, time, threading, signal
from datetime import datetime
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info

FAUCET_PORT  = 6653
FAULT_LOG    = "faults_fattree16.log"
RND_INTERVAL = (10, 60)
RND_DURATION = (5, 25)

class FatTree4(Topo):
    """k=4 fat-tree: 4 core, 4 aggregation, 8 edge switches, 16 hosts."""
    def build(self):
        k = 4
        # Core switches: k^2/4 = 4
        core = [self.addSwitch(f'c{i+1}') for i in range(k*k//4)]
        # Pods
        agg, edge = [], []
        host_id = 1
        for pod in range(k):
            pod_agg  = [self.addSwitch(f'a{pod+1}{j+1}') for j in range(k//2)]
            pod_edge = [self.addSwitch(f'e{pod+1}{j+1}') for j in range(k//2)]
            agg.extend(pod_agg); edge.extend(pod_edge)
            # Core ↔ Aggregation
            for j, a in enumerate(pod_agg):
                for ci in range(k//2):
                    self.addLink(core[j*(k//2)+ci], a)
            # Aggregation ↔ Edge
            for a in pod_agg:
                for e in pod_edge:
                    self.addLink(a, e)
            # Hosts on edge switches (k/2 hosts per edge)
            for e in pod_edge:
                for _ in range(k//2):
                    h = self.addHost(f'h{host_id}',
                                     ip=f'10.{pod+1}.{host_id}.1/24')
                    self.addLink(h, e)
                    host_id += 1

def log_fault(tag, msg, logfile=FAULT_LOG):
    ts   = datetime.utcnow().isoformat()
    line = f"{ts} {tag} {msg}"
    print(line)
    with open(logfile, 'a') as f:
        f.write(line + '\n')

def fault_injector(net):
    while True:
        time.sleep(random.randint(*RND_INTERVAL))
        h = random.choice(net.hosts)
        intf = h.defaultIntf().name
        log_fault('START', f'LINKDOWN intf={intf}')
        h.defaultIntf().ifconfig('down')
        start = datetime.utcnow()
        time.sleep(random.randint(*RND_DURATION))
        h.defaultIntf().ifconfig('up')
        log_fault('END',   f'LINKUP intf={intf}')
        dur = (datetime.utcnow() - start).total_seconds()
        log_fault('SUMMARY', f'LINK_FAULT duration={dur:.1f}s')

def main():
    setLogLevel('info')
    os.system('mn -c')
    topo = FatTree4()
    net  = Mininet(topo=topo, link=TCLink,
                   controller=lambda name: RemoteController(
                       name, ip='127.0.0.1', port=FAUCET_PORT),
                   autoSetMacs=True)
    net.start()
    threading.Thread(target=fault_injector, args=(net,), daemon=True).start()
    info("Fat-tree-16 topology running. Ctrl-D to quit.\n")
    CLI(net)
    net.stop()

if __name__ == '__main__':
    main()