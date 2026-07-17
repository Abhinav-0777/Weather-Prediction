import os
from functools import lru_cache
from dotenv import load_dotenv

@lru_cache
def get_env():
    load_dotenv('.env')
    return dict(os.environ)