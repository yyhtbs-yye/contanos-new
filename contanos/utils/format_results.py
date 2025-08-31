def format_to_list(results):
    """Format detection results by converting array-like objects to Python lists."""
    if results is None:
        return {'results': {}}
    
    def convert_to_list(obj):
        """Convert various array-like objects to Python lists."""
        if obj is None:
            return []
        
        # Handle empty containers
        if hasattr(obj, '__len__') and len(obj) == 0:
            return []
        
        # Convert numpy arrays, tensors, or other array-like objects
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        
        # Handle list of array-like objects (e.g., list of numpy arrays)
        if isinstance(obj, (list, tuple)):
            converted_list = []
            for item in obj:
                if hasattr(item, 'tolist'):
                    converted_list.append(item.tolist())
                elif isinstance(item, (int, float, str, bool)):
                    converted_list.append(item)
                else:
                    # Try to convert to appropriate type
                    try:
                        if isinstance(item, (list, tuple)):
                            converted_list.append(list(item))
                        else:
                            converted_list.append(item)
                    except:
                        converted_list.append(str(item))
            return converted_list
        
        # Handle scalar values
        if isinstance(obj, (int, float, str, bool)):
            return obj
        
        # Try to convert to list as fallback
        try:
            return list(obj)
        except:
            return obj
    
    # Process all keys in the results dictionary
    formatted_results = {}
    for key, value in results.items():
        if key == 'img':
            formatted_results[key] = value
        else:
            formatted_results[key] = convert_to_list(value)
    
    return {'results': formatted_results}