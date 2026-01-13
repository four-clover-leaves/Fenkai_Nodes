import os
import configparser
from typing import Optional

# Attempt to load dotenv for local .env support
try:
    from dotenv import load_dotenv
    load_dotenv() 
except ImportError:
    pass

def load_gemini_api_key() -> Optional[str]:
    """
    Centralized API key loader.
    Priority: 1. Environment (API_GOOGLE) | 2. Local .ini
    """
    # 1. Check Environment (ComfyDeploy Cloud or local .env)
    env_key = os.environ.get("API_GOOGLE")
    if env_key:
        print("--- Nano Banana Pro: Using API_GOOGLE from Environment ---")
        return env_key

    # 2. Legacy .ini file check (Local fallback)
    node_dir = os.path.dirname(__file__)
    node_ini = os.path.join(node_dir, "googleapi.ini")
    
    if os.path.exists(node_ini):
        cfg = configparser.ConfigParser()
        try:
            cfg.read(node_ini)
            local_key = cfg.get("API_KEY", "key", fallback=None)
            if local_key:
                print("--- Nano Banana Pro: Using Local .ini Key ---")
                return local_key
        except Exception:
            pass
            
    return None