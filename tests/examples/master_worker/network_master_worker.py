def flatten_packets(packets):
    # Extrai um relatório por pacote válido (um item de saída por item de entrada)
    reports = []
    for pkt in packets:
        if pkt["valid"]:
            reports.append(f"{pkt['protocol']}: {pkt['size']} bytes")
    return reports


if __name__ == "__main__":
    packets = [
        {"valid": True,  "size": 1500, "protocol": "TCP"},
        {"valid": False, "size": 1200, "protocol": "UDP"},
        {"valid": True,  "size": 800,  "protocol": "ICMP"},
        {"valid": True,  "size": 2000, "protocol": "TCP"},
    ]
    # esperado: ['TCP: 1500 bytes', 'ICMP: 800 bytes', 'TCP: 2000 bytes']
    print(flatten_packets(packets))
