# setup.py
import subprocess
import sys
import os
import time

def print_step(step: str):
    print(f"\n{'─'*50}")
    print(f"  {step}")
    print('─'*50)

def print_ok(msg: str):
    print(f"  ✅ {msg}")

def print_fail(msg: str):
    print(f"  ❌ {msg}")

def print_info(msg: str):
    print(f"  ℹ️  {msg}")

def check_python():
    print_step("Checking Python version")
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print_ok(f"Python {version.major}.{version.minor}.{version.micro}")
    else:
        print_fail(f"Python 3.11+ required, found {version.major}.{version.minor}")
        sys.exit(1)

def check_docker():
    print_step("Checking Docker")
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print_ok(result.stdout.strip())
        else:
            raise Exception()
    except Exception:
        print_fail("Docker not found")
        print_info("Install Docker Desktop from https://docker.com/products/docker-desktop")
        sys.exit(1)

    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print_ok(result.stdout.strip())
        else:
            raise Exception()
    except Exception:
        print_fail("Docker Compose not found — update Docker Desktop")
        sys.exit(1)

def start_postgres():
    print_step("Starting PostgreSQL with pgvector (Docker)")
    result = subprocess.run(
        ["docker", "compose", "up", "-d", "postgres"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print_fail(f"Failed to start PostgreSQL: {result.stderr}")
        sys.exit(1)
    print_ok("PostgreSQL container started")

    # wait for health check to pass
    print_info("Waiting for PostgreSQL to be ready...")
    for attempt in range(30):
        result = subprocess.run(
            ["docker", "compose", "exec", "postgres",
             "pg_isready", "-U", "rag_user", "-d", "rag_db"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print_ok("PostgreSQL is ready and accepting connections")
            return
        time.sleep(2)
        print(f"  waiting... ({attempt + 1}/30)")

    print_fail("PostgreSQL did not become ready in time")
    sys.exit(1)

def check_env():
    print_step("Checking environment variables")
    from dotenv import load_dotenv
    load_dotenv()

    required = ["GENERATION_MODEL", "LOCAL_DB_URL"]
    optional = ["TAVILY_API_KEY"]
    all_good = True

    for var in required:
        val = os.getenv(var)
        if val:
            print_ok(f"{var} is set")
        else:
            print_fail(f"{var} is not set — add it to your .env file")
            all_good = False

    for var in optional:
        val = os.getenv(var)
        if val:
            print_ok(f"{var} is set (optional)")
        else:
            print_info(f"{var} not set — web search will use DuckDuckGo fallback")

    if not all_good:
        print_info("Copy .env.example to .env and fill in your values")
        sys.exit(1)

def check_db_connection():
    print_step("Checking database connection")
    try:
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv()
        conn = psycopg2.connect(os.getenv("LOCAL_DB_URL"))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM documents")
        count = cur.fetchone()[0]
        conn.close()
        print_ok(f"Connected to database ({count} documents in knowledge base)")
    except Exception as e:
        print_fail(f"Database connection failed: {e}")
        print_info("Make sure Docker is running and LOCAL_DB_URL matches docker-compose.yml credentials")
        sys.exit(1)

def check_lm_studio():
    print_step("Checking LM Studio connection")
    try:
        import requests
        from dotenv import load_dotenv
        load_dotenv()
        base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
        response = requests.get(f"{base_url}/models", timeout=5)
        if response.status_code == 200:
            models = response.json().get("data", [])
            if models:
                print_ok(f"LM Studio is running with {len(models)} model(s) loaded")
                for m in models:
                    print_info(f"  Model: {m.get('id', 'unknown')}")
            else:
                print_fail("LM Studio is running but no model is loaded")
                print_info("Load a model in LM Studio and start the server")
        else:
            raise Exception(f"Status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print_fail("Cannot connect to LM Studio")
        print_info(f"Make sure LM Studio is running with the server started at {base_url}")
        sys.exit(1)
    except Exception as e:
        print_fail(f"LM Studio check failed: {e}")
        sys.exit(1)

def check_hardware_tier():
    print_step("Hardware tier")
    from dotenv import load_dotenv
    load_dotenv()
    tier = os.getenv("HARDWARE_TIER", "large")
    tiers = {
        "small":  "7B models  | 8-16GB RAM  | conservative settings",
        "medium": "13B models | 16-32GB RAM | balanced settings",
        "large":  "30B+ models | 32GB+ RAM  | full quality settings",
    }
    if tier in tiers:
        print_ok(f"Tier: {tier} — {tiers[tier]}")
    else:
        print_fail(f"Unknown tier '{tier}' — use small, medium, or large")
        sys.exit(1)

def run_quick_test():
    print_step("Running quick end-to-end test")
    try:
        from pipeline import ingest, query

        print_info("Ingesting test document...")
        result = ingest(
            "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
        )
        if result.get("skipped"):
            print_ok("Test document already ingested (skipped)")
        else:
            print_ok(f"Ingested {result['ingested']} chunks")

        print_info("Running test query...")
        result = query("What is RAG?")
        if result.get("answer"):
            print_ok(f"Pipeline working — route: {result['route']}")
            print_info(f"Answer preview: {result['answer'][:150]}...")
        else:
            print_fail("Pipeline returned empty answer")

    except Exception as e:
        print_fail(f"End-to-end test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("\n🔧 RAG Pipeline Setup\n")
    print("This script validates your environment and starts required services.")

    check_python()
    check_docker()
    start_postgres()
    check_env()
    check_db_connection()
    check_lm_studio()
    check_hardware_tier()
    run_quick_test()

    print(f"\n{'='*50}")
    print("  ✅ Setup complete — pipeline is ready")
    print('='*50)
    print("\nNext steps:")
    print("  Ingest content:  python -c \"from pipeline import ingest; ingest('https://...')\"")
    print("  Query pipeline:  python -c \"from pipeline import query; print(query('your question')['answer'])\"")
    print("  Start MCP server: python mcp_server.py")
    print("  Run eval suite:  python -m eval.ragas_runner\n")