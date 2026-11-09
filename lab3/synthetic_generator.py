"""
Synthetic NetFlow Traffic Generator for Cybersecurity Anomaly Detection.
Part A of Lab 03: ML/DL Techniques in Cybersecurity.
Author: Cybersecurity Student / AI Lab 3
"""

import numpy as np
import pandas as pd
import argparse
import os

def generate_synthetic_netflow(n_flows: int = 60000, anomaly_rate: float = 0.06, random_state: int = 42) -> pd.DataFrame:
    """
    Generates realistic NetFlow-like statistical network flows with normal profiles
    and injected cybersecurity attack anomalies.

    Parameters:
    -----------
    n_flows : int
        Total number of flows to generate (must be >= 50,000).
    anomaly_rate : float
        Proportion of anomalous flows in the dataset (e.g., 0.01, 0.05, 0.10).
    random_state : int
        Seed for reproducibility.

    Returns:
    --------
    pd.DataFrame:
        DataFrame containing network flow statistical features, detailed profile labels,
        and binary anomaly labels.
    """
    rng = np.random.default_rng(random_state)
    
    n_anomalies = int(n_flows * anomaly_rate)
    n_normal = n_flows - n_anomalies
    
    # Normal traffic profiles distribution:
    # 1. Web browsing: 40%
    # 2. DNS requests: 25%
    # 3. Legitimate SSH: 10%
    # 4. Downloads / Updates: 15%
    # 5. File transfers / Backups: 10%
    norm_counts = {
        'web_browsing': int(n_normal * 0.40),
        'dns_query': int(n_normal * 0.25),
        'ssh_session': int(n_normal * 0.10),
        'software_update': int(n_normal * 0.15),
        'file_backup': n_normal - (int(n_normal * 0.40) + int(n_normal * 0.25) + int(n_normal * 0.10) + int(n_normal * 0.15))
    }
    
    # Anomalous traffic scenarios distribution:
    # 1. Port scanning: 25%
    # 2. DDoS: 25%
    # 3. Data exfiltration: 25%
    # 4. Brute force: 25%
    anom_counts = {
        'port_scan': int(n_anomalies * 0.25),
        'ddos_flood': int(n_anomalies * 0.25),
        'data_exfiltration': int(n_anomalies * 0.25),
        'brute_force': n_anomalies - 3 * int(n_anomalies * 0.25)
    }

    flows = []

    # =========================================================================
    # NORMAL PROFILES (A.1)
    # =========================================================================
    
    # 1. Web Browsing (HTTP/HTTPS)
    cnt = norm_counts['web_browsing']
    # HTTP (80) or HTTPS (443)
    dst_ports = rng.choice([80, 443, 8080], size=cnt, p=[0.20, 0.75, 0.05])
    protocols = np.full(cnt, 6) # TCP
    # Duration: lognormal (typical 0.2s to 30s)
    durations = np.clip(rng.lognormal(mean=0.8, sigma=0.8, size=cnt), 0.05, 60.0)
    # Packets: client requests + server responses (10 to 80 packets)
    src_packets = np.clip(rng.negative_binomial(10, 0.5, size=cnt) + 4, 3, 100)
    dst_packets = np.clip(rng.negative_binomial(15, 0.35, size=cnt) + 5, 4, 250)
    # Bytes: Asymmetric (more incoming than outgoing)
    src_bytes = np.clip(rng.lognormal(mean=6.8, sigma=0.6, size=cnt), 200, 15000)
    dst_bytes = np.clip(rng.lognormal(mean=9.5, sigma=1.1, size=cnt), 500, 300000)
    # Window dynamics
    conn_window = np.clip(rng.poisson(lam=12, size=cnt), 1, 40)
    unique_ports = np.clip(rng.choice([1, 2, 3], size=cnt, p=[0.7, 0.25, 0.05]), 1, 5)
    inter_arrival = np.clip(rng.exponential(scale=1.2, size=cnt), 0.01, 30.0)
    
    for i in range(cnt):
        flows.append({
            'duration': durations[i],
            'protocol': protocols[i],
            'dst_port': dst_ports[i],
            'src_bytes': src_bytes[i],
            'dst_bytes': dst_bytes[i],
            'src_packets': src_packets[i],
            'dst_packets': dst_packets[i],
            'conn_count_window': conn_window[i],
            'unique_ports_window': unique_ports[i],
            'inter_arrival_time': inter_arrival[i],
            'traffic_type': 'web_browsing',
            'label': 0
        })

    # 2. DNS Requests
    cnt = norm_counts['dns_query']
    dst_ports = np.full(cnt, 53)
    protocols = rng.choice([17, 6], size=cnt, p=[0.98, 0.02]) # Mostly UDP
    # Duration: sub-second (query and quick response)
    durations = np.clip(rng.lognormal(mean=-3.8, sigma=0.6, size=cnt), 0.002, 0.25)
    src_packets = np.clip(rng.choice([1, 2, 3], size=cnt, p=[0.75, 0.20, 0.05]), 1, 5)
    dst_packets = np.clip(rng.choice([1, 2, 3], size=cnt, p=[0.80, 0.15, 0.05]), 1, 5)
    src_bytes = np.clip(rng.normal(loc=75, scale=18, size=cnt), 40, 250)
    dst_bytes = np.clip(rng.normal(loc=160, scale=45, size=cnt), 60, 800)
    conn_window = np.clip(rng.poisson(lam=5, size=cnt), 1, 25)
    unique_ports = np.ones(cnt, dtype=int)
    inter_arrival = np.clip(rng.exponential(scale=2.5, size=cnt), 0.005, 45.0)

    for i in range(cnt):
        flows.append({
            'duration': durations[i],
            'protocol': protocols[i],
            'dst_port': dst_ports[i],
            'src_bytes': src_bytes[i],
            'dst_bytes': dst_bytes[i],
            'src_packets': src_packets[i],
            'dst_packets': dst_packets[i],
            'conn_count_window': conn_window[i],
            'unique_ports_window': unique_ports[i],
            'inter_arrival_time': inter_arrival[i],
            'traffic_type': 'dns_query',
            'label': 0
        })

    # 3. Legitimate SSH Connections
    cnt = norm_counts['ssh_session']
    dst_ports = np.full(cnt, 22)
    protocols = np.full(cnt, 6) # TCP
    # Duration: prolonged interactive session
    durations = np.clip(rng.lognormal(mean=4.2, sigma=0.9, size=cnt), 10.0, 900.0)
    src_packets = np.clip(rng.lognormal(mean=4.8, sigma=0.7, size=cnt), 20, 1500).astype(int)
    dst_packets = np.clip(rng.lognormal(mean=5.1, sigma=0.7, size=cnt), 25, 2000).astype(int)
    # Low-throughput keystrokes and light command output
    src_bytes = np.clip(rng.lognormal(mean=8.0, sigma=0.8, size=cnt), 1500, 60000)
    dst_bytes = np.clip(rng.lognormal(mean=8.9, sigma=0.9, size=cnt), 3000, 120000)
    conn_window = np.clip(rng.choice([1, 2, 3], size=cnt, p=[0.75, 0.20, 0.05]), 1, 5)
    unique_ports = np.ones(cnt, dtype=int)
    inter_arrival = np.clip(rng.exponential(scale=15.0, size=cnt), 0.5, 120.0)

    for i in range(cnt):
        flows.append({
            'duration': durations[i],
            'protocol': protocols[i],
            'dst_port': dst_ports[i],
            'src_bytes': src_bytes[i],
            'dst_bytes': dst_bytes[i],
            'src_packets': src_packets[i],
            'dst_packets': dst_packets[i],
            'conn_count_window': conn_window[i],
            'unique_ports_window': unique_ports[i],
            'inter_arrival_time': inter_arrival[i],
            'traffic_type': 'ssh_session',
            'label': 0
        })

    # 4. Downloads and Software Updates
    cnt = norm_counts['software_update']
    dst_ports = rng.choice([80, 443, 8080], size=cnt, p=[0.25, 0.70, 0.05])
    protocols = np.full(cnt, 6) # TCP
    # Duration: moderate to long
    durations = np.clip(rng.lognormal(mean=3.2, sigma=0.8, size=cnt), 5.0, 300.0)
    # High packets, heavy inbound payload
    dst_bytes = np.clip(rng.lognormal(mean=14.0, sigma=1.2, size=cnt), 50000, 25000000) # up to 25 MB
    dst_packets = np.clip(dst_bytes / rng.uniform(1200, 1460, size=cnt), 50, 20000).astype(int)
    src_packets = np.clip(dst_packets * rng.uniform(0.4, 0.6, size=cnt), 25, 12000).astype(int)
    src_bytes = np.clip(src_packets * rng.uniform(50, 75, size=cnt), 1500, 800000) # TCP ACKs
    conn_window = np.clip(rng.poisson(lam=4, size=cnt), 1, 15)
    unique_ports = np.clip(rng.choice([1, 2], size=cnt, p=[0.85, 0.15]), 1, 3)
    inter_arrival = np.clip(rng.exponential(scale=6.0, size=cnt), 0.1, 45.0)

    for i in range(cnt):
        flows.append({
            'duration': durations[i],
            'protocol': protocols[i],
            'dst_port': dst_ports[i],
            'src_bytes': src_bytes[i],
            'dst_bytes': dst_bytes[i],
            'src_packets': src_packets[i],
            'dst_packets': dst_packets[i],
            'conn_count_window': conn_window[i],
            'unique_ports_window': unique_ports[i],
            'inter_arrival_time': inter_arrival[i],
            'traffic_type': 'software_update',
            'label': 0
        })

    # 5. File Transfers and Backups
    cnt = norm_counts['file_backup']
    dst_ports = rng.choice([445, 2049, 21, 22], size=cnt, p=[0.50, 0.20, 0.15, 0.15])
    protocols = np.full(cnt, 6) # TCP
    durations = np.clip(rng.lognormal(mean=3.8, sigma=1.0, size=cnt), 8.0, 600.0)
    # Heavy outbound volume (or balanced sync)
    src_bytes = np.clip(rng.lognormal(mean=14.2, sigma=1.2, size=cnt), 60000, 30000000) # up to 30MB
    src_packets = np.clip(src_bytes / rng.uniform(1200, 1460, size=cnt), 50, 22000).astype(int)
    dst_packets = np.clip(src_packets * rng.uniform(0.4, 0.6, size=cnt), 25, 12000).astype(int)
    dst_bytes = np.clip(dst_packets * rng.uniform(50, 80, size=cnt), 2000, 900000)
    conn_window = np.clip(rng.poisson(lam=3, size=cnt), 1, 10)
    unique_ports = np.ones(cnt, dtype=int)
    inter_arrival = np.clip(rng.exponential(scale=10.0, size=cnt), 0.2, 60.0)

    for i in range(cnt):
        flows.append({
            'duration': durations[i],
            'protocol': protocols[i],
            'dst_port': dst_ports[i],
            'src_bytes': src_bytes[i],
            'dst_bytes': dst_bytes[i],
            'src_packets': src_packets[i],
            'dst_packets': dst_packets[i],
            'conn_count_window': conn_window[i],
            'unique_ports_window': unique_ports[i],
            'inter_arrival_time': inter_arrival[i],
            'traffic_type': 'file_backup',
            'label': 0
        })

    # =========================================================================
    # ANOMALY INJECTION (A.2)
    # Note: Deliberately introduce overlapping regions with normal traffic
    # to avoid artificial, trivial linear separability.
    # =========================================================================

    # 1. Port Scanning
    cnt = anom_counts['port_scan']
    # Scanner touches many distinct ports, some common, some random ephemeral
    common_scan_ports = [21, 22, 23, 25, 53, 80, 110, 139, 443, 445, 1433, 3306, 3389, 8080, 8443]
    random_high_ports = rng.integers(1024, 65535, size=cnt)
    dst_ports = np.where(rng.random(size=cnt) < 0.6, rng.choice(common_scan_ports, size=cnt), random_high_ports)
    protocols = rng.choice([6, 17], size=cnt, p=[0.85, 0.15])
    # Very short connection attempts
    durations = np.clip(rng.exponential(scale=0.03, size=cnt), 0.001, 1.2)
    # Minimal packets (SYN sent, RST received or timeout)
    # Some stealth scans overlap with normal brief probes (1-4 pkts)
    src_packets = rng.choice([1, 2, 3], size=cnt, p=[0.7, 0.25, 0.05])
    dst_packets = rng.choice([0, 1, 2], size=cnt, p=[0.5, 0.45, 0.05])
    src_bytes = src_packets * rng.integers(40, 65, size=cnt)
    dst_bytes = dst_packets * rng.integers(40, 65, size=cnt)
    # SCAN SIGNATURE: Abnormally high unique ports in recent window!
    # Overlap note: Slow stealth scans have lower port counts (6-15) overlapping with busy web browsers
    unique_ports = np.clip(rng.integers(15, 180, size=cnt), 5, 250)
    conn_window = unique_ports + rng.integers(0, 20, size=cnt)
    inter_arrival = np.clip(rng.exponential(scale=0.08, size=cnt), 0.001, 2.0)

    for i in range(cnt):
        flows.append({
            'duration': durations[i],
            'protocol': protocols[i],
            'dst_port': dst_ports[i],
            'src_bytes': src_bytes[i],
            'dst_bytes': dst_bytes[i],
            'src_packets': src_packets[i],
            'dst_packets': dst_packets[i],
            'conn_count_window': conn_window[i],
            'unique_ports_window': unique_ports[i],
            'inter_arrival_time': inter_arrival[i],
            'traffic_type': 'port_scan',
            'label': 1
        })

    # 2. DDoS (Distributed Denial of Service)
    cnt = anom_counts['ddos_flood']
    # Floods usually target web (80, 443), DNS (53), or game/app ports
    dst_ports = rng.choice([80, 443, 53, 8080], size=cnt, p=[0.45, 0.35, 0.15, 0.05])
    protocols = rng.choice([6, 17], size=cnt, p=[0.70, 0.30])
    # Short burst flows
    durations = np.clip(rng.exponential(scale=0.15, size=cnt), 0.002, 3.5)
    src_packets = rng.integers(2, 25, size=cnt)
    dst_packets = np.where(rng.random(size=cnt) < 0.6, rng.integers(0, 3, size=cnt), rng.integers(2, 10, size=cnt))
    src_bytes = src_packets * rng.integers(60, 400, size=cnt)
    dst_bytes = dst_packets * rng.integers(40, 300, size=cnt)
    # DDOS SIGNATURE: Massive surge in connection counts over window, microscopic inter-arrival times
    # Overlap note: Lower DDoS rates (40-60) overlap with peak web page multi-connection bursts
    conn_window = np.clip(rng.integers(60, 600, size=cnt), 30, 800)
    unique_ports = rng.choice([1, 2], size=cnt, p=[0.90, 0.10])
    inter_arrival = np.clip(rng.exponential(scale=0.008, size=cnt), 0.0001, 0.05)

    for i in range(cnt):
        flows.append({
            'duration': durations[i],
            'protocol': protocols[i],
            'dst_port': dst_ports[i],
            'src_bytes': src_bytes[i],
            'dst_bytes': dst_bytes[i],
            'src_packets': src_packets[i],
            'dst_packets': dst_packets[i],
            'conn_count_window': conn_window[i],
            'unique_ports_window': unique_ports[i],
            'inter_arrival_time': inter_arrival[i],
            'traffic_type': 'ddos_flood',
            'label': 1
        })

    # 3. Data Exfiltration
    cnt = anom_counts['data_exfiltration']
    # Exfiltration often mimics HTTP/HTTPS or DNS tunneling
    dst_ports = rng.choice([443, 80, 8080, 53], size=cnt, p=[0.55, 0.30, 0.10, 0.05])
    protocols = np.where(dst_ports == 53, 17, 6)
    # Long duration sustained outbound transfer
    durations = np.clip(rng.lognormal(mean=4.0, sigma=0.8, size=cnt), 15.0, 700.0)
    # MASSIVE src_bytes sent OUTWARDS compared to normal browsing/DNS!
    # Overlap note: Heavy exfiltration volumes overlap directly with legitimate file_backup!
    # The anomaly is detected through the unusual port/protocol combination (e.g. 15MB sent over HTTPS or DNS)
    src_bytes = np.clip(rng.lognormal(mean=14.8, sigma=0.9, size=cnt), 150000, 35000000) # 150KB to 35MB
    src_packets = np.clip(src_bytes / rng.uniform(1100, 1460, size=cnt), 100, 26000).astype(int)
    dst_packets = np.clip(src_packets * rng.uniform(0.35, 0.55, size=cnt), 40, 14000).astype(int)
    dst_bytes = np.clip(dst_packets * rng.uniform(40, 85, size=cnt), 2000, 1000000)
    conn_window = np.clip(rng.poisson(lam=2, size=cnt), 1, 8)
    unique_ports = np.ones(cnt, dtype=int)
    inter_arrival = np.clip(rng.exponential(scale=8.0, size=cnt), 0.1, 50.0)

    for i in range(cnt):
        flows.append({
            'duration': durations[i],
            'protocol': protocols[i],
            'dst_port': dst_ports[i],
            'src_bytes': src_bytes[i],
            'dst_bytes': dst_bytes[i],
            'src_packets': src_packets[i],
            'dst_packets': dst_packets[i],
            'conn_count_window': conn_window[i],
            'unique_ports_window': unique_ports[i],
            'inter_arrival_time': inter_arrival[i],
            'traffic_type': 'data_exfiltration',
            'label': 1
        })

    # 4. Brute Force
    cnt = anom_counts['brute_force']
    # Repeated attempts against SSH, FTP, RDP, or HTTP login
    dst_ports = rng.choice([22, 21, 3389, 80, 443], size=cnt, p=[0.45, 0.25, 0.15, 0.10, 0.05])
    protocols = np.full(cnt, 6) # TCP
    # Regular short authentication attempts
    durations = np.clip(rng.normal(loc=1.2, scale=0.35, size=cnt), 0.2, 4.0)
    # Uniform packet/byte signature of failed credential exchange
    # Overlap note: An individual brute force flow looks almost identical to a legitimate quick SSH or FTP login!
    src_packets = np.clip(rng.normal(loc=9, scale=2.5, size=cnt), 4, 20).astype(int)
    dst_packets = np.clip(rng.normal(loc=11, scale=3.0, size=cnt), 5, 25).astype(int)
    src_bytes = np.clip(src_packets * rng.uniform(70, 110, size=cnt), 300, 2500)
    dst_bytes = np.clip(dst_packets * rng.uniform(80, 130, size=cnt), 400, 3500)
    # BRUTE FORCE SIGNATURE: High connection count to the *same* target port, rapid repeated retries
    conn_window = np.clip(rng.integers(25, 120, size=cnt), 15, 150)
    unique_ports = np.ones(cnt, dtype=int)
    inter_arrival = np.clip(rng.exponential(scale=0.4, size=cnt), 0.02, 3.0)

    for i in range(cnt):
        flows.append({
            'duration': durations[i],
            'protocol': protocols[i],
            'dst_port': dst_ports[i],
            'src_bytes': src_bytes[i],
            'dst_bytes': dst_bytes[i],
            'src_packets': src_packets[i],
            'dst_packets': dst_packets[i],
            'conn_count_window': conn_window[i],
            'unique_ports_window': unique_ports[i],
            'inter_arrival_time': inter_arrival[i],
            'traffic_type': 'brute_force',
            'label': 1
        })

    df = pd.DataFrame(flows)
    
    # Compute derived NetFlow features
    df['total_bytes'] = df['src_bytes'] + df['dst_bytes']
    df['total_packets'] = df['src_packets'] + df['dst_packets']
    df['bytes_per_second'] = df['total_bytes'] / (df['duration'] + 1e-4)
    df['packets_per_second'] = df['total_packets'] / (df['duration'] + 1e-4)
    df['bytes_per_packet'] = df['total_bytes'] / np.maximum(df['total_packets'], 1)
    df['byte_ratio'] = (df['src_bytes'] + 1.0) / (df['dst_bytes'] + 1.0)

    # Shuffle the dataset so flows are interleaved realistically
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic NetFlow network traffic.")
    parser.add_argument('--flows', type=int, default=60000, help="Total number of flows (default: 60000 >= 50000)")
    parser.add_argument('--anomaly-rate', type=float, default=0.06, help="Anomaly rate (default: 0.06 or 6%%)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--output', type=str, default='synthetic_traffic.csv', help="Output CSV path")
    args = parser.parse_args()

    print(f"Generating {args.flows} flows with anomaly rate {args.anomaly_rate:.2%} (seed={args.seed})...")
    df = generate_synthetic_netflow(n_flows=args.flows, anomaly_rate=args.anomaly_rate, random_state=args.seed)
    
    print("\nDataset Summary:")
    print(f"Total flows: {len(df):,}")
    print("Class distribution:")
    print(df['label'].value_counts(normalize=True).map(lambda x: f"{x:.2%}"))
    print("\nDetailed profile breakdown:")
    print(df['traffic_type'].value_counts())
    
    df.to_csv(args.output, index=False)
    print(f"\nSaved successfully to {args.output} ({os.path.getsize(args.output)/1e6:.2f} MB).")
