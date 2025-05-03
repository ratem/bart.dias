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
