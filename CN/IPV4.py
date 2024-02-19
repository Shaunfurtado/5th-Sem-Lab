def classify_and_identify_ipv4(ip_address : str):
    octets = ip_address.split('.')
    
    if len(octets) != 4:
        return "Invalid Ipv4 address"
    
    first_octet = int(octets[0])
    
    if 1 <= first_octet <= 127:
        classification = "Class A"
        network_id = octets[0]
        host_id = '.'.join(octets[1:])
    elif 128 <= first_octet <= 191:
        classification = "Class B"
        network_id = '.'.join(octets[:2])
        host_id = '.'.join(octets[2:])
    elif 192 <= first_octet <= 223:
        classification = "Class C"
        network_id = '.'.join(octets[:3])
        host_id = '.'.join(octets[3:])
    elif 224 <= first_octet <= 239:
        classification = "Class D"
        network_id = "N/A"
        host_id = "N/A"        
    elif 240 <= first_octet <= 255:
        classification = "Class E"
        network_id = "N/A"
        host_id = "N/A"
    else:
        return "Invalid IPv4 address"
    
    return f"Classification: {classification} \n Network ID: {network_id}\n Host ID: {host_id}"


user_input = input("Enter an IPv4 address: ")
results = classify_and_identify_ipv4(user_input)
print(results)