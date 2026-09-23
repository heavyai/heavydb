<!--
SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Integration tests

Docker-based integration tests for HeavyDB client features that need a multi-process environment to run (database server, message broker, TLS, client libraries).

Each test lives in its own subdirectory under `dev-tools/integration-tests/` while its Docker Compose definitions live under `docker/dev-*`. For example docker/dev-kafka-import-test/docker-compose.yml.

Tests can be run via **`dev-tools/dev.sh`** or directly via each test’s `run.sh` script with a required **`--build-dir`** argument: your CMake binary directory (`CMAKE_BINARY_DIR`), which must contain **`bin/heavydb`**. The repo source path is derived from the script location (`dev-tools/integration-tests/<test>/` → three levels up).

```bash
dev-tools/dev.sh test integration-encrypted-jdbc --build-dir build
dev-tools/dev.sh test integration-kafka-import --build-dir build
dev-tools/integration-tests/encrypted-jdbc/run.sh --build-dir build
dev-tools/integration-tests/kafka-import/run.sh --build-dir build
```

Nested build directories work the same way, for example `--build-dir xb` or `--build-dir build/release`.

Note — build artifacts from a prior CMake build are required before running any test (see **Build dependencies** below).

The `run.sh` script performs the following general actions:

1. Sets **`HEAVYDB_BUILD`** from **`--build-dir`** and **`HEAVYDB_SOURCE`** from the repo layout, then exports container names and other test-specific settings.
2. Starts long-running services with **`docker compose up -d --wait`**. Docker runs **healthchecks** on those services and Compose blocks (on a timer) until the containers report they are healthy.
3. Starts one-shot **setup** and **verify** test containers when `depends_on` conditions are met (`service_healthy` or `service_completed_successfully`). These containers run the test clients.
4. Tears the stack down on exit (including on failure).

**HeavyDB storage** defaults to **`/tmp/heavydb-data` inside the `heavydb` container** (under `/tmp`, writable with `USER_SPEC`) — nothing is bind-mounted to the host, so catalog/data are removed when the container is removed. For debugging, pass **`--host-storage`** (or set **`USE_HOST_STORAGE=1`**) to merge `docker-compose.host-storage.yml`, which bind-mounts a per-test directory under `storage/` on the host at **`/heavydb-data`**. With host storage, `run.sh` creates that directory before `compose up` to avoid file permissions issues.

All traffic stays on the **Docker Compose network**. There are **no host port mappings**, so tests do not conflict with a local HeavyDB install on 6274/6278/6279.

Scripts invoked inside containers live in the scripts directory next to each test’s `run.sh` and are bind-mounted via the repo root (`/heavydb`). Compose files in `docker/` only describe services, volumes, healthchecks, and entrypoints.

**Requirements**

- Docker Compose **v2.29+** (`up --wait`, `service_completed_successfully`)
- A prior CMake build whose binary directory is passed as **`--build-dir`** (see **Build dependencies** below)
- NVIDIA GPU available in Docker for HeavyDB services (compose `deploy.resources.reservations.devices`)
- Default container images (override with `--image` / `--kafka-image` in `run.sh`):
  - **HeavyDB** (all HeavyDB services): `ghcr.io/heavyai/heavydb/core-build-ubuntu22.04-static-cuda12.9.2-x86_64:latest`
  - **Kafka** (kafka-import test only): `apache/kafka:3.7.2`
- For encrypted JDBC: Maven dependency cache — see **Maven cache (`~/.m2`)** under the encrypted JDBC test below.

## Build dependencies

These tests do **not** compile HeavyDB inside Docker. They bind-mount your CMake binary directory (`--build-dir`, exposed in containers as `/heavydb-build`) and, for JDBC verification, the repo source tree (`HEAVYDB_SOURCE` → `/heavydb`). You must build the required targets on the host **before** running a test.

`run.sh` only checks that **`bin/heavydb`** exists and is executable; other missing artifacts surface as runtime failures inside the containers.

### C++ binaries under `--build-dir/bin/`

| Artifact | Used by | Role |
| --- | --- | --- |
| `heavydb` | all tests | Database server |
| `initheavy` | all tests | Initialize catalog/storage before `heavydb` starts |
| `heavysql` | all tests | SQL client for setup, healthchecks, and verification |
| `KafkaImporter` | kafka-import only | Consumes Kafka messages and loads the `flights` table |

### Java / Calcite (all tests that start `heavydb`)

`heavydb` always launches the Calcite Java server during startup (`DBHandler` constructs a `Calcite` instance, which spawns the JAR). Both integration tests start `heavydb`, so they need the Calcite artifacts below regardless of encryption.

| Artifact | Location | Role |
| --- | --- | --- |
| `calcite-1.41.0-SNAPSHOT-jar-with-dependencies.jar` | `--build-dir/bin/` | Calcite server JAR launched by `heavydb` (path derived from the `heavydb` executable location) |
| `QueryEngine/` | `--build-dir/QueryEngine/` | Extension path passed to Calcite (`-e`) |

The build declares `heavydb` and `initheavy` as depending on the **`mapd_java_components`** target, which runs the Thrift compiler to generate the Java sources and builds this JAR.

### Thrift Java sources and JDBC client build (encrypted JDBC only)

The encrypted JDBC test additionally compiles and runs JDBC tests from the bind-mounted **`java/`** tree with Maven. That step does **not** re-run Thrift codegen.

| Artifact | Location | Role |
| --- | --- | --- |
| Thrift-generated Java sources | `java/thrift/src/gen/` in the **source tree** | Types such as `ai.heavy.thrift.server.*` required to compile the `thrift` Maven module and downstream JDBC code |

These sources are **generated by the build process** (from `.thrift` IDL such as `heavy.thrift` and `java/thrift/calciteserver.thrift`, via the Thrift compiler invoked from the build) and are **gitignored** — they are not present after a fresh clone until a build has run. If `java/thrift/src/gen/` is empty, the JDBC verify step fails with errors like `package ai.heavy.thrift.server does not exist`.

Maven third-party dependencies (Apache Thrift, JUnit, etc.) are resolved at test time from the host **`~/.m2`** cache (or an isolated empty volume with `--empty-m2`); those are not produced by the CMake build.

## Layout

```
dev-tools/
  integration-tests/
    README.md
    generate-calcite-test-keystore.sh   # shared TLS material generator
    kafka-import/                       # KafkaImporter test
    encrypted-jdbc/                     # TLS + JDBC client test

docker/
  dev-kafka-import-test/
    docker-compose.yml
    docker-compose.host-storage.yml   # optional; merged with --host-storage
  dev-encrypted-jdbc/
    docker-compose.yml
    docker-compose.host-storage.yml   # optional; merged with --host-storage
    docker-compose.empty-m2.yml       # optional; merged with --empty-m2
```

---

## Kafka import test

**Directory:** `kafka-import/`
**Compose:** `docker/dev-kafka-import-test/`

### What it tests

End-to-end **KafkaImporter** flow against an unencrypted HeavyDB instance:

1. Initialize storage and start HeavyDB.
2. Create a `flights` table (schema matches `SampleData/100_flights.csv`).
3. Create a Kafka topic and produce 100 CSV data rows (header stripped).
4. Run **KafkaImporter**, poll until 100 rows appear in `flights`.

Dataset: `SampleData/100_flights.csv`.

### How to run

```text
dev-tools/dev.sh test integration-kafka-import --build-dir build
dev-tools/integration-tests/kafka-import/run.sh --build-dir build
    [--host-storage --image ghcr.io/heavyai/heavydb/core-build-ubuntu22.04-static-cuda12.9.2-x86_64:latest --kafka-image apache/kafka:3.7.2]
```

Bracketed flags are optional. See `run.sh --help` for defaults.

## Encrypted JDBC test

**Directory:** `encrypted-jdbc/`
**Compose:** `docker/dev-encrypted-jdbc/`

### What it tests

HeavyDB with **TLS** enabled (Thrift, HTTP, Calcite), then two verification steps:

1. **Backend encryption** — `heavysql` over TLS (`:6274`) runs `SHOW TABLES` and checks default `initheavy` geo sample tables (`heavyai_us_states`, `heavyai_countries`).
2. **JDBC over HTTPS** — Maven runs `HeavyAIConnectionTest` methods **tst3a–tst3d** (four combinations of truststore and hostname verification) against `heavydb:6278`.

TLS material is generated at startup into a named compose volume (`/heavydb-ca-info` in containers) by `generate-calcite-test-keystore.sh` (also used standalone for dev keystores). The TLS volume is removed when `run.sh` exits.

### How to run

```text
dev-tools/dev.sh test integration-encrypted-jdbc --build-dir build
dev-tools/integration-tests/encrypted-jdbc/run.sh --build-dir build
    [--host-storage --empty-m2 --image ghcr.io/heavyai/heavydb/core-build-ubuntu22.04-static-cuda12.9.2-x86_64:latest]
```

Bracketed flags are optional. See `run.sh --help` for defaults.

### Maven cache (`~/.m2`)

The JDBC verify step runs **Maven** inside the `verify-jdbc` container. Java sources are built from the bind-mounted repo tree (`HEAVYDB_SOURCE/java`); third-party jars are resolved into a local Maven repository.

**Default (no `--empty-m2`):** `run.sh` bind-mounts the invoking user’s host **`~/.m2`** into the container at **`/m2`**. Maven uses **`/m2/repository`**, which is the same cache as on the host.

This sharing has two practical effects:

- **Faster re-runs** — dependencies downloaded or built in a previous test (or a normal host `mvn` run) are reused, so later runs avoid re-downloading the same artifacts.
- **Side effects on the host cache** — the test may **write new artifacts** into `~/.m2/repository` (for example when dependencies change, such as a Thrift version bump). Those files **remain on the host** after the test exits. In rare cases, directories created inside the container with the wrong ownership can leave **root-owned paths** under `~/.m2/repository` that block later runs; ensure `~/.m2/repository` is writable by your user.

**Isolated cache (`--empty-m2`):** merges `docker-compose.empty-m2.yml`, which mounts an **empty, container-only** named volume at `/m2/repository` instead of the host `~/.m2`. Maven still downloads everything it needs, but **nothing is written to the host Maven cache**. The volume is removed when `run.sh` exits. Use this for a clean-room dependency fetch, CI-style isolation, or to avoid touching a shared `~/.m2`. The first run will be slower because the cache starts empty.

If Maven fails immediately resolving `project-settings-extension`, a stale `.lastUpdated` marker under `~/.m2` from a prior failed download may be blocking retries — see `scripts/ci/build.sh` for how CI avoids this bootstrap issue.

## Shared utilities

### `generate-calcite-test-keystore.sh`

Builds dev TLS assets (CA, server certs, Java truststore, `heavyai-tls.conf`) under a configurable output directory. Encrypted JDBC writes to a named compose volume mounted at `/heavydb-ca-info`; `ca-info-init` chowns that volume for `USER_SPEC` before `heavydb` starts. `run.sh` removes the TLS volume on exit.

Can be run directly for manual TLS setup; the encrypted JDBC `entrypoint.sh` invokes it automatically when the test starts.

## Future improvements

### Reduce duplication in scripts and compose

Much of the host orchestration is repeated across the two `run.sh` files (compose detection, teardown trap, `--host-storage` merge, build-directory checks, and storage preparation). A shared script under `dev-tools/integration-tests/shared/` (for example `run-common.sh`, `container-init.sh`, and a common `env.sh`) would be the highest-value refactor. In-container scripts share the same mapd-deps bootstrap and logging boilerplate; those could be centralised too. Compose files repeat the HeavyDB GPU block, healthcheck shape, and repo/build bind mounts — some of that could be shared via Compose `include` if more tests are added, though YAML merge rules (especially for host-storage overrides) limit how far consolidation can go without hurting clarity. Test-specific orchestration and entrypoints should stay in each test’s own directory.

### Candidates for additional integration tests

Each new test would follow the same pattern: a subdirectory under `dev-tools/integration-tests/`, a matching `docker/dev-*` compose stack, and a `run.sh` with **`--build-dir`**.

- **General JDBC tests (non-encrypted)** — exercise `HeavyAIConnectionTest` over plain Thrift/HTTP against an unencrypted HeavyDB (similar to encrypted-jdbc’s `docker-jdbc-runner.sh`, but without TLS material or HTTPS-only cases).
- **SQLImporter test** — end-to-end import of a file-based dataset via SQLImporter (analogous to kafka-import’s KafkaImporter flow: start HeavyDB, stage input data, run the importer, verify row counts).
- **Encrypted backend-only smoke test** — optional split of encrypted-jdbc’s `verify-backend-encryption` if JDBC and backend checks should run independently in CI.
