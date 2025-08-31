def parse_config_string(config_string):
    """Parse connection string into dict with addr and topic keys"""
    config = {}
    
    # Split by comma and parse key=value pairs
    parts = config_string.split(',')
    base_url = parts[0]  # First part is the base URL
    
    # Extract address from URL (protocol://host:port)
    config['addr'] = base_url
    
    # Parse remaining parameters
    for part in parts[1:]:
        if '=' in part:
            key, value = part.split('=', 1)
            config[key] = value
    
    return config
