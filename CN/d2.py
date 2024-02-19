import socket

# Take input for IPv4 header fields
ttl = int(input("Enter TTL (Time To Live): "))
source_ip = input("Enter Source IP address: ")
destination_ip = input("Enter Destination IP address: ")

# TCP header fields
source_port = int(input("Enter Source Port: "))
destination_port = int(input("Enter Destination Port: "))

# IPv4 header fields
version = 4
header_length = 5
protocol = 6  # TCP protocol

# Constructing the IPv4 packet
ipv4_header = bytearray()
ipv4_header += ((version << 4) + header_length).to_bytes(1, 'big')
ipv4_header += ttl.to_bytes(1, 'big')
ipv4_header += protocol.to_bytes(1, 'big')
ipv4_header += socket.inet_aton(source_ip)
ipv4_header += socket.inet_aton(destination_ip)

# TCP header
tcp_header = bytearray()
tcp_header += source_port.to_bytes(2, 'big')
tcp_header += destination_port.to_bytes(2, 'big')
tcp_header += b'\x00\x00\x00\x00'  # Sequence number (4 bytes)
tcp_header += b'\x00\x00\x00\x00'  # Acknowledgment number (4 bytes)
tcp_header += b'\x50\x02'  # Data offset, Reserved, and Flags (2 bytes)
tcp_header += b'\xff\xff'  # Window size (2 bytes)
tcp_header += b'\x00\x00'  # Checksum (2 bytes)
tcp_header += b'\x00\x00'  # Urgent pointer (2 bytes)

# Displaying the constructed IPv4 packet
print("IPv4 Packet:")
print("Version:", version)
print("Header Length:", header_length)
print("TTL:", ttl)
print("Protocol:", protocol)
print("Source IP:", source_ip)
print("Destination IP:", destination_ip)
print("Raw Bytes (IPv4 header):", ipv4_header.hex())
print("Raw Bytes (TCP header):", tcp_header.hex())