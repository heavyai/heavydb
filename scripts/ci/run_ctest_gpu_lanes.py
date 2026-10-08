#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the CTest suite concurrently in isolated single-GPU lanes.

Ordinary tests receive one visible GPU, a private catalog, and private Calcite
ports. ExecuteTest is sharded across the lanes. Tests that exercise peer access
or multi-device behavior run serially at the end with all visible GPUs.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


EXECUTE_TEST_NAMES = (
    "ExecuteTest",
    "ExecuteTestTemporaryTables",
    "ExecuteTestExecutorResourceMgr",
)

MULTI_GPU_CTEST_NAMES = (
    "CudaMgrTest",
    "GpuSharedMemoryTest",
    "GroupByTest",
    "MultiInstanceTest",
)

MULTI_GPU_EXECUTE_TESTS = (
    "Select.PerDeviceCardinality",
    "Select.PerDeviceCardinalityShardedTable",
    "Select.SortWithCPUQueryHint",
    "Select.DeferredGpuMultiStorageAggregateConsumerMaterializesColumns",
    "Select.PipelinedGpuReductionMergesKeylessPerfectHashAggregates",
    "Select.PipelinedGpuReductionPreservesBaselineHashAverageSlots",
    "Select.PipelinedGpuReductionMergesLargePerfectHashInputsAsPeerTree",
    "Select.PipelinedGpuReductionMergesNonKeylessPerfectHashGroupKeys",
    "Select.PayloadFreeRankedBitmapAllreducePreservesGlobalMembership",
    "Select.BaselineBoundaryAppendReducesSplitFragmentKeys",
)


@dataclasses.dataclass(frozen=True)
class TestDefinition:
    name: str
    command: tuple[str, ...]
    working_directory: Path
    cost: float


@dataclasses.dataclass(frozen=True)
class Task:
    name: str
    test: TestDefinition
    environment: dict[str, str] = dataclasses.field(default_factory=dict)
    reset_catalog: bool = False
    use_lane_layout: bool = True


@dataclasses.dataclass(frozen=True)
class Lane:
    number: int
    root: Path
    gpu: str
    calcite_port: int
    db_handler_calcite_port: int

    @property
    def test_directory(self) -> Path:
        return self.root / "Tests"


@dataclasses.dataclass(frozen=True)
class TaskResult:
    task: Task
    lane: int
    return_code: int
    duration_seconds: float
    log_path: Path
    xml_path: Path | None
    timed_out: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default="build", type=Path)
    parser.add_argument("--jobs", type=int, help="Maximum number of GPU lanes")
    parser.add_argument(
        "--execute-shards",
        type=int,
        help="ExecuteTest shard count (default: number of lanes)",
    )
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--keep-lanes", action="store_true")
    return parser.parse_args()


def run_checked(command: list[str], *, cwd: Path, environment: dict[str, str]) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}{result.stderr}"
        )
    return result.stdout


def discover_gpus(environment: dict[str, str]) -> list[str]:
    configured = environment.get("CUDA_VISIBLE_DEVICES")
    if configured is not None:
        devices = [item.strip() for item in configured.split(",") if item.strip()]
        if devices == ["-1"]:
            return []
        return devices

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError(f"Unable to enumerate GPUs: {result.stderr.strip()}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def read_costs(build_directory: Path) -> dict[str, float]:
    cost_file = build_directory / "Testing/Temporary/CTestCostData.txt"
    costs: dict[str, float] = {}
    if not cost_file.is_file():
        return costs
    for line in cost_file.read_text().splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0] != "---":
            try:
                costs[fields[0]] = float(fields[2])
            except ValueError:
                pass
    return costs


def discover_tests(
    build_directory: Path, environment: dict[str, str]
) -> list[TestDefinition]:
    raw = json.loads(
        run_checked(
            ["ctest", "--test-dir", str(build_directory), "--show-only=json-v1"],
            cwd=build_directory.parent,
            environment=environment,
        )
    )
    costs = read_costs(build_directory)
    tests: list[TestDefinition] = []
    for test in raw["tests"]:
        properties = {item["name"]: item["value"] for item in test.get("properties", [])}
        working_directory = Path(properties.get("WORKING_DIRECTORY", build_directory))
        tests.append(
            TestDefinition(
                name=test["name"],
                command=tuple(test["command"]),
                working_directory=working_directory,
                cost=costs.get(test["name"], 0.0),
            )
        )
    return tests


def allocate_ports(count: int) -> list[int]:
    sockets: list[socket.socket] = []
    ports: list[int] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
            ports.append(sock.getsockname()[1])
    finally:
        for sock in sockets:
            sock.close()
    return ports


def symlink(source: Path, destination: Path) -> None:
    destination.symlink_to(source, target_is_directory=source.is_dir())


def initialize_catalog(
    initheavy: Path, test_directory: Path, environment: dict[str, str]
) -> None:
    destination = test_directory / "tmp"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    run_checked(
        [str(initheavy), "-f", "./tmp"],
        cwd=test_directory,
        environment=environment,
    )


def prepare_lane(
    *,
    number: int,
    gpu: str,
    ports: tuple[int, int],
    repo_root: Path,
    build_directory: Path,
    lane_base: Path,
    results_directory: Path,
    environment: dict[str, str],
) -> Lane:
    lane_directory = lane_base / f"lane-{number}"
    root = lane_directory / "build"
    test_directory = root / "Tests"
    test_directory.mkdir(parents=True)

    # Several tests intentionally create fixtures through paths such as
    # ../../Tests/Import/datafiles. Give each lane its own source-test tree so
    # those writers cannot remove another lane's input while it is in use.
    shutil.copytree(repo_root / "Tests", lane_directory / "Tests", symlinks=True)
    for source_name in ("QueryEngine", "QueryRenderer"):
        source = repo_root / source_name
        if source.exists():
            symlink(source, lane_directory / source_name)

    for entry in build_directory.iterdir():
        if entry.name == "Tests" or entry.name.startswith(".gpu-test-lanes-"):
            continue
        if entry.is_dir():
            symlink(entry, root / entry.name)

    # Execute binaries through lane-local links. Some tests derive storage and
    # temporary-file paths from argv[0], which must agree with the private CWD.
    # Copy non-executables because sqliteTestDB is a writable reference database;
    # sharing that inode across lanes causes lock conflicts and cross-shard mutation.
    for entry in (build_directory / "Tests").iterdir():
        if entry.is_file():
            destination = test_directory / entry.name
            if entry.stat().st_mode & 0o111:
                symlink(entry, destination)
            else:
                shutil.copy2(entry, destination)
    symlink(build_directory / "bin/initheavy", root / "initheavy")
    symlink(results_directory, root / "test-results")
    # Catalogs contain path-dependent system-server options. Initialize each one
    # at its final path rather than copying a catalog created elsewhere.
    initialize_catalog(build_directory / "bin/initheavy", test_directory, environment)

    return Lane(number, root, gpu, ports[0], ports[1])


def prepare_lanes(
    *,
    repo_root: Path,
    build_directory: Path,
    gpus: list[str],
    results_directory: Path,
    environment: dict[str, str],
) -> tuple[Path, list[Lane], Lane]:
    lane_base = build_directory / f".gpu-test-lanes-{os.getpid()}"
    lane_base.mkdir()
    ports = allocate_ports((len(gpus) + 1) * 2)
    lanes = [
        prepare_lane(
            number=index,
            gpu=gpu,
            ports=(ports[index * 2], ports[index * 2 + 1]),
            repo_root=repo_root,
            build_directory=build_directory,
            lane_base=lane_base,
            results_directory=results_directory,
            environment=environment,
        )
        for index, gpu in enumerate(gpus)
    ]
    multi_index = len(gpus)
    multi_lane = prepare_lane(
        number=multi_index,
        gpu=",".join(gpus),
        ports=(ports[multi_index * 2], ports[multi_index * 2 + 1]),
        repo_root=repo_root,
        build_directory=build_directory,
        lane_base=lane_base,
        results_directory=results_directory,
        environment=environment,
    )
    return lane_base, lanes, multi_lane


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def make_tasks(
    tests: list[TestDefinition], execute_shards: int
) -> tuple[list[Task], list[Task]]:
    by_name = {test.name: test for test in tests}
    missing = set(EXECUTE_TEST_NAMES + MULTI_GPU_CTEST_NAMES) - set(by_name)
    if missing:
        raise RuntimeError(f"Required CTest targets are missing: {sorted(missing)}")

    single_tasks: list[Task] = []
    excluded_filter = "-" + ":".join(MULTI_GPU_EXECUTE_TESTS)
    for test_name in EXECUTE_TEST_NAMES:
        test = by_name[test_name]
        for shard in range(execute_shards):
            single_tasks.append(
                Task(
                    name=f"{test_name}-shard-{shard:02d}-of-{execute_shards:02d}",
                    test=test,
                    environment={
                        "GTEST_TOTAL_SHARDS": str(execute_shards),
                        "GTEST_SHARD_INDEX": str(shard),
                        "GTEST_FILTER": excluded_filter,
                    },
                )
            )

    excluded_ctest = set(EXECUTE_TEST_NAMES + MULTI_GPU_CTEST_NAMES)
    ordinary = sorted(
        (test for test in tests if test.name not in excluded_ctest),
        key=lambda test: (-test.cost, test.name),
    )
    single_tasks.extend(
        Task(
            name=test.name,
            test=test,
            # SessionsStoreTest expects a complete catalog but does not recreate one
            # after an earlier test removes the lane-local tmp directory.
            reset_catalog=test.name == "SessionsStoreTest",
        )
        for test in ordinary
    )

    multi_filter = ":".join(MULTI_GPU_EXECUTE_TESTS)
    multi_tasks = [
        Task(
            name=f"{test_name}-multi-gpu",
            test=by_name[test_name],
            environment={"GTEST_FILTER": multi_filter},
            reset_catalog=True,
        )
        for test_name in EXECUTE_TEST_NAMES
    ]
    multi_tasks.extend(
        Task(
            name=name,
            test=by_name[name],
            reset_catalog=name != "MultiInstanceTest",
            use_lane_layout=name != "MultiInstanceTest",
        )
        for name in MULTI_GPU_CTEST_NAMES
    )
    return single_tasks, multi_tasks


def rewrite_command(
    task: Task, lane: Lane, results_directory: Path
) -> tuple[list[str], Path | None]:
    command = list(task.test.command)
    output_path: Path | None = None
    output_name = safe_name(task.name) + ".xml"

    if task.use_lane_layout:
        original_executable = Path(command[0])
        if original_executable.parent.name == "Tests" and original_executable.is_file():
            lane_executable = lane.test_directory / original_executable.name
            if lane_executable.is_file():
                command[0] = str(lane_executable)

    for index, argument in enumerate(command):
        if argument.startswith("--gtest_output="):
            output_path = results_directory / output_name
            command[index] = f"--gtest_output=xml:{output_path}"
        elif task.use_lane_layout and argument.startswith("BUILD_DIR="):
            command[index] = f"BUILD_DIR={lane.root}"

    if any(argument.startswith("BINARY=Tests/ArrowIpcIntegrationTest") for argument in command):
        output_path = results_directory / output_name
        command.insert(-1, f"RESULTS_FILE=test-results/{output_name}")

    return command, output_path


def tail(path: Path, line_count: int = 30) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-line_count:])
    except OSError:
        return ""


def run_task(
    *,
    task: Task,
    lane: Lane,
    results_directory: Path,
    base_environment: dict[str, str],
    timeout: float,
    output_lock: threading.Lock,
) -> TaskResult:
    if task.reset_catalog:
        initialize_catalog(lane.root / "initheavy", lane.test_directory, base_environment)

    environment = dict(base_environment)
    for variable in (
        "GTEST_FILTER",
        "GTEST_SHARD_INDEX",
        "GTEST_SHARD_STATUS_FILE",
        "GTEST_TOTAL_SHARDS",
    ):
        environment.pop(variable, None)
    environment.update(task.environment)
    environment["CUDA_VISIBLE_DEVICES"] = lane.gpu
    environment["HEAVYDB_TEST_CALCITE_PORT"] = str(lane.calcite_port)
    environment["HEAVYDB_TEST_DB_HANDLER_CALCITE_PORT"] = str(
        lane.db_handler_calcite_port
    )
    environment.setdefault("CMAKE_PREFIX_PATH", "")
    environment.setdefault("LD_LIBRARY_PATH", "")

    command, xml_path = rewrite_command(task, lane, results_directory)
    working_directory = (
        lane.test_directory if task.use_lane_layout else task.test.working_directory
    )
    if "GTEST_SHARD_INDEX" in environment:
        environment["GTEST_SHARD_STATUS_FILE"] = str(
            results_directory / (safe_name(task.name) + ".shard-status")
        )

    log_path = results_directory / (safe_name(task.name) + ".log")
    with output_lock:
        print(
            f"START lane={lane.number} gpu={lane.gpu} task={task.name}",
            flush=True,
        )

    started = time.monotonic()
    timed_out = False
    with log_path.open("w") as log_file:
        log_file.write(f"cwd: {working_directory}\n")
        log_file.write(f"command: {' '.join(command)}\n")
        log_file.write(f"CUDA_VISIBLE_DEVICES={lane.gpu}\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                return_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                return_code = process.wait()

    duration = time.monotonic() - started
    status = "PASS" if return_code == 0 and not timed_out else "FAIL"
    with output_lock:
        print(
            f"{status} lane={lane.number} task={task.name} seconds={duration:.2f}",
            flush=True,
        )
        if status == "FAIL":
            print(tail(log_path), flush=True)
    return TaskResult(
        task=task,
        lane=lane.number,
        return_code=return_code,
        duration_seconds=duration,
        log_path=log_path,
        xml_path=xml_path,
        timed_out=timed_out,
    )


def run_parallel_tasks(
    *,
    tasks: list[Task],
    lanes: list[Lane],
    results_directory: Path,
    environment: dict[str, str],
    timeout: float,
) -> list[TaskResult]:
    task_queue: queue.Queue[Task] = queue.Queue()
    for task in tasks:
        task_queue.put(task)

    results: list[TaskResult] = []
    results_lock = threading.Lock()
    output_lock = threading.Lock()

    def worker(lane: Lane) -> None:
        while True:
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                return
            started = time.monotonic()
            try:
                result = run_task(
                    task=task,
                    lane=lane,
                    results_directory=results_directory,
                    base_environment=environment,
                    timeout=timeout,
                    output_lock=output_lock,
                )
            except Exception:
                log_path = results_directory / (safe_name(task.name) + ".log")
                log_path.write_text(traceback.format_exc())
                result = TaskResult(
                    task=task,
                    lane=lane.number,
                    return_code=1,
                    duration_seconds=time.monotonic() - started,
                    log_path=log_path,
                    xml_path=None,
                )
                with output_lock:
                    print(
                        f"FAIL lane={lane.number} task={task.name} "
                        f"seconds={result.duration_seconds:.2f}",
                        flush=True,
                    )
                    print(tail(log_path), flush=True)
            finally:
                task_queue.task_done()
            with results_lock:
                results.append(result)

    threads = [threading.Thread(target=worker, args=(lane,)) for lane in lanes]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def parse_xml_results(results: list[TaskResult]) -> dict[str, int]:
    totals = {"tests": 0, "failures": 0, "disabled": 0, "skipped": 0, "missing": 0}
    for result in results:
        if result.xml_path is None:
            continue
        if not result.xml_path.is_file():
            totals["missing"] += 1
            continue
        root = ET.parse(result.xml_path).getroot()
        totals["tests"] += int(root.attrib.get("tests", 0))
        totals["failures"] += int(root.attrib.get("failures", 0))
        totals["disabled"] += int(root.attrib.get("disabled", 0))
        totals["skipped"] += sum(
            testcase.attrib.get("result") == "skipped"
            for testcase in root.findall(".//testcase")
        )
    return totals


def write_summary(
    path: Path,
    *,
    started_at: str,
    wall_seconds: float,
    gpus: list[str],
    results: list[TaskResult],
    xml_totals: dict[str, int],
) -> None:
    payload: dict[str, Any] = {
        "started_at": started_at,
        "wall_seconds": wall_seconds,
        "gpus": gpus,
        "xml_totals": xml_totals,
        "tasks": [
            {
                "name": result.task.name,
                "ctest_name": result.task.test.name,
                "lane": result.lane,
                "return_code": result.return_code,
                "duration_seconds": result.duration_seconds,
                "timed_out": result.timed_out,
                "log": str(result.log_path),
                "xml": str(result.xml_path) if result.xml_path else None,
            }
            for result in sorted(results, key=lambda item: item.task.name)
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    build_directory = args.build_dir.resolve()
    if not (build_directory / "CTestTestfile.cmake").is_file():
        raise RuntimeError(f"Not a configured CTest build: {build_directory}")

    environment = dict(os.environ)
    environment.setdefault("CMAKE_PREFIX_PATH", "")
    environment.setdefault("LD_LIBRARY_PATH", "")
    gpus = discover_gpus(environment)
    if not gpus:
        raise RuntimeError("No visible GPUs; use the normal serial CTest runner")
    if args.jobs is not None:
        if args.jobs < 1:
            raise RuntimeError("--jobs must be positive")
        gpus = gpus[: args.jobs]
    execute_shards = args.execute_shards or len(gpus)
    if execute_shards < 1:
        raise RuntimeError("--execute-shards must be positive")

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_directory = (
        args.results_dir.resolve()
        if args.results_dir
        else build_directory / "Testing/GpuLanes" / timestamp
    )
    results_directory.mkdir(parents=True)

    tests = discover_tests(build_directory, environment)
    single_tasks, multi_tasks = make_tasks(tests, execute_shards)
    print(
        f"Detected {len(gpus)} GPU(s); {len(single_tasks)} lane task(s), "
        f"{len(multi_tasks)} multi-GPU tail task(s)",
        flush=True,
    )

    # Compute this before lane preparation so a partially created layout is still
    # removed if catalog initialization fails.
    lane_base: Path | None = build_directory / f".gpu-test-lanes-{os.getpid()}"
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    started = time.monotonic()
    results: list[TaskResult] = []
    try:
        lane_base, lanes, multi_lane = prepare_lanes(
            repo_root=repo_root,
            build_directory=build_directory,
            gpus=gpus,
            results_directory=results_directory,
            environment=environment,
        )
        results.extend(
            run_parallel_tasks(
                tasks=single_tasks,
                lanes=lanes,
                results_directory=results_directory,
                environment=environment,
                timeout=args.timeout,
            )
        )
        print("Single-GPU lanes drained; starting exclusive multi-GPU tail", flush=True)
        output_lock = threading.Lock()
        for task in multi_tasks:
            results.append(
                run_task(
                    task=task,
                    lane=multi_lane,
                    results_directory=results_directory,
                    base_environment=environment,
                    timeout=args.timeout,
                    output_lock=output_lock,
                )
            )
    finally:
        if lane_base and lane_base.exists() and not args.keep_lanes:
            shutil.rmtree(lane_base)

    wall_seconds = time.monotonic() - started
    xml_totals = parse_xml_results(results)
    failures = [result for result in results if result.return_code or result.timed_out]
    write_summary(
        results_directory / "summary.json",
        started_at=started_at,
        wall_seconds=wall_seconds,
        gpus=gpus,
        results=results,
        xml_totals=xml_totals,
    )
    print(
        f"SUMMARY tasks={len(results)} failures={len(failures)} "
        f"wall_seconds={wall_seconds:.2f} xml={xml_totals} "
        f"results={results_directory}",
        flush=True,
    )
    return 1 if failures or xml_totals["failures"] or xml_totals["missing"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
