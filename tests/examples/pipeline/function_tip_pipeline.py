def analyze_network_traffic(packets):
    # Stage 1: Packet filtering
    filtered = []
    for pkt in packets:
        if pkt['valid']:
            filtered.append(pkt)

    # Stage 2: Traffic analysis
    stats = []
    for pkt in filtered:
        stats.append({
            'size': pkt['size'],
            'type': pkt['protocol']
        })

    # Stage 3: Report generation
    reports = []
    for stat in stats:
        reports.append(f"{stat['type']}: {stat['size']} bytes")

    return reports

# print(analyze_network_traffic([
#     {'valid': True, 'size': 1500, 'protocol': 'TCP'},
#     {'valid': False, 'size': 1200, 'protocol': 'UDP'},
#     {'valid': True, 'size': 800, 'protocol': 'ICMP'},
#     {'valid': True, 'size': 2000, 'protocol': 'TCP'}
# ]))