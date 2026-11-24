"""Start a local PaperAgent test environment on Windows, macOS or Linux."""
import argparse
import math
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "paperagent.config.json"
EXAMPLE = ROOT / "paperagent.config.example.json"
JWT_KEY_FILE = ROOT / ".paperagent-jwt-key"


def local_jwt_key(path=JWT_KEY_FILE):
    """Persist the local signing key so unexpired tokens survive a restart."""
    if path.exists():
        key = path.read_text(encoding="utf-8").strip()
        if not key:
            raise RuntimeError(f"Local JWT key file is empty: {path}")
        return key
    fd, temporary = tempfile.mkstemp(prefix=".paperagent-jwt-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(secrets.token_urlsafe(48))
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        key = path.read_text(encoding="utf-8").strip()
        if not key:
            raise RuntimeError(f"Local JWT key file is empty: {path}")
        return key
    finally:
        os.unlink(temporary)


def check_port(port):
    with socket.socket() as sock:
        sock.settimeout(0.5)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} is already in use. Stop the existing service and retry.")


def wait_for_http(url, process, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Service exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)
    raise RuntimeError(f"Service did not respond within {timeout} seconds: {url}")


def ensure_redis(url):
    import redis

    client = redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
    try:
        client.ping()
        return None
    except redis.RedisError as exc:
        if url not in ("redis://127.0.0.1:6379/0", "redis://localhost:6379/0") or not shutil.which("docker"):
            raise RuntimeError("Redis is unavailable. Start Redis and check redis_url in the configuration.") from exc
    name = "paperagent-local-" + secrets.token_hex(4)
    result = subprocess.run(["docker", "run", "-d", "--rm", "--name", name, "-p", "127.0.0.1:6379:6379", "-v", "paperagent-redis-data:/data", "redis:7", "redis-server", "--appendonly", "yes"],
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError("Could not start Redis with Docker. Start Redis manually and retry.")
    try:
        for _ in range(40):
            try:
                client.ping()
                return name
            except redis.RedisError:
                time.sleep(0.25)
        raise RuntimeError("Redis container started but did not respond")
    except BaseException:
        subprocess.run(["docker", "stop", name], capture_output=True, check=False)
        raise


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main():
    parser = argparse.ArgumentParser(description="Start the PaperAgent local test environment")
    parser.add_argument("--setup", action="store_true", help="install Python and frontend dependencies, then exit")
    parser.add_argument("--check", action="store_true", help="run Python tests and frontend build, then exit")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.setup:
        npm = shutil.which("npm")
        if not npm:
            raise RuntimeError("Install Node.js and npm before running --setup")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[web,papers,dev]", "fakeredis[lua]"],
                       check=True, cwd=ROOT)
        subprocess.run([npm, "ci"], check=True, cwd=ROOT / "web")
        return
    node = shutil.which("node")
    vite = ROOT / "web" / "node_modules" / "vite" / "bin" / "vite.js"
    if not node or not vite.is_file():
        raise RuntimeError("Install Node.js, then run: cd web && npm ci")
    if args.check:
        from corecoder.runtime_config import FIELDS

        test_env = os.environ.copy()
        for name in FIELDS.values():
            test_env.pop(name, None)
        test_env.pop("PAPER_CONFIG_PATH", None)
        test_env["PAPER_DISABLE_CONFIG"] = "1"
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "-rs"], check=True, cwd=ROOT, env=test_env)
        subprocess.run([node, "--test", "tests/workflowSync.test.mjs"], check=True, cwd=ROOT / "web")
        vue_tsc = ROOT / "web" / "node_modules" / "vue-tsc" / "bin" / "vue-tsc.js"
        subprocess.run([node, str(vue_tsc), "--noEmit"], check=True, cwd=ROOT / "web")
        subprocess.run([node, str(vite), "build"], check=True, cwd=ROOT / "web")
        return
    if not CONFIG.exists():
        # Exclusive creation preserves an existing file if two launchers start together.
        with CONFIG.open("x", encoding="utf-8") as output:
            output.write(EXAMPLE.read_text(encoding="utf-8"))
        print(f"Created {CONFIG}. Fill in an API key and model paths, then run this command again.")
        return
    import jwt

    from corecoder.runtime_config import apply_runtime_config

    apply_runtime_config(CONFIG)
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("Set deepseek_api_key or openai_api_key in paperagent.config.json")
    size = int(os.environ.get("PAPER_CHUNK_SIZE", "1024"))
    overlap = int(os.environ.get("PAPER_CHUNK_OVERLAP", "50"))
    threshold = float(os.environ.get("PAPER_RERANK_THRESHOLD", "0.3"))
    if size <= 0 or not 0 <= overlap < size or not 0 <= threshold <= 1:
        raise ValueError("Check chunk_size, chunk_overlap and rerank_threshold in the configuration")
    if int(os.environ.get("PAPER_RETRIEVAL_LIMIT", "5")) < 1 or int(os.environ.get("PAPER_RETRIEVAL_CANDIDATES", "30")) < 1:
        raise ValueError("retrieval_limit and retrieval_candidates must be positive")
    timeout = float(os.environ.get("PAPER_LLM_TIMEOUT", "60"))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("llm_timeout_seconds must be positive and finite")
    retrieval_timeout = float(os.environ.get("PAPER_RETRIEVAL_TIMEOUT", "180"))
    if not math.isfinite(retrieval_timeout) or retrieval_timeout <= 0:
        raise ValueError("retrieval_timeout_seconds must be positive and finite")
    check_port(8000)
    check_port(5173)
    redis_name = None
    backend = None
    frontend = None
    try:
        redis_name = ensure_redis(os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0"))
        os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6379/0")
        if "CORECODER_JWT_KEY" not in os.environ:
            os.environ["CORECODER_JWT_KEY"] = local_jwt_key()
        os.environ.setdefault("CORECODER_JWT_ISSUER", "paperagent-local")
        os.environ.setdefault("CORECODER_JWT_AUDIENCE", "paperagent-web")
        now = int(time.time())
        token = jwt.encode({"sub": "local-user", "iss": os.environ["CORECODER_JWT_ISSUER"],
                            "aud": os.environ["CORECODER_JWT_AUDIENCE"], "iat": now, "exp": now + 86400},
                           os.environ["CORECODER_JWT_KEY"], algorithm="HS256")
        bootstrap_code = secrets.token_urlsafe(32)
        os.environ["PAPER_LOCAL_BOOTSTRAP_CODE"] = bootstrap_code
        os.environ["PAPER_LOCAL_BOOTSTRAP_EXPIRES"] = str(time.time() + 600)
        backend = subprocess.Popen([sys.executable, "-m", "uvicorn", "corecoder.web:app", "--host", "127.0.0.1",
                                    "--port", "8000"], cwd=ROOT)
        wait_for_http("http://127.0.0.1:8000/api/health", backend)
        frontend = subprocess.Popen([node, str(vite), "--host", "127.0.0.1", "--port", "5173", "--strictPort"],
                                    cwd=ROOT / "web")
        wait_for_http("http://127.0.0.1:5173/", frontend)
        login_url = "http://127.0.0.1:5173/#local-session=" + bootstrap_code
        print("\nPaperAgent one-time sign-in link (valid for 10 minutes):", flush=True)
        print(login_url, flush=True)
        print("Alternatively, paste this local test token into the access token field:", flush=True)
        print(token, flush=True)
        print("Select the provider matching your API key. Press Ctrl+C to stop.", flush=True)
        webbrowser.open(login_url)
        while backend.poll() is None and frontend.poll() is None:
            time.sleep(0.5)
        raise RuntimeError("A service exited unexpectedly")
    finally:
        stop_process(frontend)
        stop_process(backend)
        if redis_name:
            subprocess.run(["docker", "stop", redis_name], capture_output=True, check=False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except (RuntimeError, ValueError, TypeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
