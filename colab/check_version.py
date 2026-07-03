# Run this IN THE COLAB notebook to verify the server code version
import os

SERVER = "orpheus-live/server/orpheus_server.py"
if not os.path.exists(SERVER):
    print(f"Server file {SERVER} not found — did cell 3 (git clone) run?")
else:
    with open(SERVER) as f:
        content = f.read()
    if "CHUNK_FRAMES = 12" in content:
        print("✓ Server has CHUNK_FRAMES = 12 (updated)")
    elif "CHUNK_FRAMES = 1" in content:
        print("✗ Server still has CHUNK_FRAMES = 1 (STALE — re-run cell 3 to git pull)")
    else:
        print("? CHUNK_FRAMES not found — check server file manually")

    if "from vllm import" in content:
        print("✓ vLLM import present")
    else:
        print("✗ vLLM import NOT found — server may be using slow transformers fallback")
