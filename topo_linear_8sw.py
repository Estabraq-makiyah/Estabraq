#!/usr/bin/env python3
"""Linear 8-switch, 16-host topology for scalability evaluation."""

import os, subprocess, random, time, threading, signal, atexit
from datetime import datetime
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info

FAUCET_PORT   = 6653
FAULT_LOG     = "faults_linear8.log"
RND_INTERVAL  = (10, 60)
RND_DURATION  = (10, 30)

class LinearTopo8(Topo):
    def build(self):
        switches = [self.addSwitch(f'sw{i}') for i in range(1, 9)]
        # Linear chain: sw1-sw2-sw3-...-sw8
        for i in range(len(switches) - 1):
            self.addLink(switches[i], switches[i+1], port1=2, port2=1)
        # 2 hosts per switch
        for i, sw in enumerate(switches):
            h1 = self.addHost(f'h{2*i+1}', ip=f'10.0.{i+1}.1/24')
            h2 = self.addHost(f'h{2*i+2}', ip=f'10.0.{i+1}.2/24')
            self.addLink(h1, sw, port1=1, port2=3)
            self.addLink(h2, sw, port1=1, port2=4)

def log_fault(tag, msg):
    ts = datetime.utcnow().isoformat()
    line = f"{ts} {tag} {msg}"
    print(line)
    with open(FAULT_LOG, 'a') as f:
        f.write(line + '\n')

def fault_injector(net):
    while True:
        time.sleep(random.randint(*RND_INTERVAL))
        kind = random.choice(['linkdown', 'porttoggle'])
        start_ts = datetime.utcnow()
        if kind == 'linkdown':
            h = random.choice(net.hosts)
            intf = h.defaultIntf().name
            log_fault('START', f'LINKDOWN intf={intf}')
            h.defaultIntf().ifconfig('down')
            time.sleep(random.randint(*RND_DURATION))
            h.defaultIntf().ifconfig('up')
            log_fault('END',   f'LINKUP intf={intf}')
        else:
            sw = random.choice(net.switches)
            p  = random.randint(1, 4)
            log_fault('START', f'PORTDOWN {sw.name}:{p}')
            subprocess.call(['ovs-ofctl', 'mod-port', sw.name, str(p), 'down'])
            time.sleep(random.randint(*RND_DURATION))
            subprocess.call(['ovs-ofctl', 'mod-port', sw.name, str(p), 'up'])
            log_fault('END',   f'PORTUP {sw.name}:{p}')
        dur = (datetime.utcnow() - start_ts).total_seconds()
        summary = {'linkdown': 'LINK_FAULT', 'porttoggle': 'PORT_FAULT'}[kind]
        log_fault('SUMMARY', f'{summary} duration={dur:.1f}s')

def main():
    setLogLevel('info')
    os.system('mn -c')
    topo = LinearTopo8()
    net  = Mininet(topo=topo, link=TCLink,
                   controller=lambda name: RemoteController(
                       name, ip='127.0.0.1', port=FAUCET_PORT),
                   autoSetMacs=True)
    net.start()
    threading.Thread(target=fault_injector, args=(net,), daemon=True).start()
    info("Linear-8 topology running. Ctrl-D to quit.\n")
    CLI(net)
    net.stop()

if __name__ == '__main__':
    main()