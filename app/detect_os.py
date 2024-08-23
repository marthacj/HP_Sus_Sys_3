import platform

def detect_os():
    system = platform.system().lower()
    
    if system == "windows":
        return "Windows"
    elif system == "darwin":
        return "macOS"
    elif system == "linux":
        return "Linux"
    else:
        return "Unknown OS"
