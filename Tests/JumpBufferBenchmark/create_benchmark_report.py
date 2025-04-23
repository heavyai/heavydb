import argparse
import dataclasses
import json
import math

from dataclasses import dataclass


@dataclass
class BenchmarkContext:
    timestamp_str: str
    hostname: str
    num_cpus: int
    gpu_mem: int
    gpu_arch: str


@dataclass
class BenchmarkRun:
    no_jump_buffer_run_time_ms: int
    with_jump_buffer_run_time_ms: int
    transfer_buffer_size: int
    jump_buffer_size: int
    parallel_copy_threads: int


@dataclass
class BenchmarkResult:
    context: BenchmarkContext
    runs: dict[str, list[BenchmarkRun]]
    statuses: dict[str, str]


def create_benchmark_run(no_jump_buffer_run: dict, with_jump_buffer_run: dict) -> BenchmarkRun:
    assert no_jump_buffer_run['jump_buffer_size'] == 0
    assert no_jump_buffer_run['parallel_copy_threads'] == 0
    assert no_jump_buffer_run['transfer_buffer_size'] == with_jump_buffer_run['transfer_buffer_size']

    return BenchmarkRun(
        no_jump_buffer_run_time_ms=no_jump_buffer_run['real_time'],
        with_jump_buffer_run_time_ms=with_jump_buffer_run['real_time'],
        transfer_buffer_size=with_jump_buffer_run['transfer_buffer_size'],
        jump_buffer_size=with_jump_buffer_run['jump_buffer_size'],
        parallel_copy_threads=with_jump_buffer_run['parallel_copy_threads'],
    )


def create_benchmark_result(file_path: str) -> BenchmarkResult:
    with (open(file_path, 'r') as file):
        result_json = json.load(file)

        context = BenchmarkContext(
            timestamp_str=result_json['context']['date'],
            hostname=result_json['context']['host_name'],
            num_cpus=result_json['context']['num_cpus'],
            gpu_mem=int(result_json['context']['gpu_mem']),
            gpu_arch=result_json['context']['gpu_arch']
        )

        benchmark_runs_dict = {}
        no_jump_buffer_run_key = 'no_jump_buffer_run'
        with_jump_buffer_run_key = 'with_jump_buffer_run'
        for benchmark in result_json['benchmarks']:
            benchmark_name_components = benchmark['name'].split('/')
            assert len(benchmark_name_components) >= 2

            benchmark_name = f'{benchmark_name_components[0]}_{benchmark_name_components[1]}'
            key = (benchmark_name, benchmark['transfer_buffer_size'])
            if key not in benchmark_runs_dict:
                benchmark_runs_dict[key] = {}
            benchmark_and_buffer_size_run_dict = benchmark_runs_dict[key]

            if benchmark['jump_buffer_size'] == 0:
                assert no_jump_buffer_run_key not in benchmark_and_buffer_size_run_dict
                benchmark_and_buffer_size_run_dict[no_jump_buffer_run_key] = benchmark
            else:
                if with_jump_buffer_run_key not in benchmark_and_buffer_size_run_dict:
                    benchmark_and_buffer_size_run_dict[with_jump_buffer_run_key] = []
                benchmark_and_buffer_size_run_dict[with_jump_buffer_run_key].append(benchmark)

        benchmark_runs = {}
        status_dict = {}
        overall_status = 'PASS'
        for key, value in benchmark_runs_dict.items():
            benchmark_name, _ = key
            if benchmark_name not in benchmark_runs:
                benchmark_runs[benchmark_name] = []
            if benchmark_name not in status_dict:
                status_dict[benchmark_name] = 'PASS'
            for run in value[with_jump_buffer_run_key]:
                benchmark_run = create_benchmark_run(value[no_jump_buffer_run_key], run)
                benchmark_runs[benchmark_name].append(benchmark_run)

                if (status_dict[benchmark_name] == 'PASS' and
                        benchmark_run.with_jump_buffer_run_time_ms > benchmark_run.no_jump_buffer_run_time_ms):
                    status_dict[benchmark_name] = 'FAIL'
                    overall_status = 'FAIL'
        status_dict['overall_status'] = overall_status
        return BenchmarkResult(context=context, runs=benchmark_runs, statuses=status_dict)


def benchmark_result_to_json_file(bench_result: BenchmarkResult, file_path: str) -> None:
    with open(file_path, 'w') as file:
        json.dump(dataclasses.asdict(bench_result), file, indent=2)
    print(f'JSON file: {file_path} created')


def benchmark_result_to_text_file(bench_result: BenchmarkResult, file_path: str) -> None:
    separator = '-' * 157
    file_content = (
        f'Timestamp: {bench_result.context.timestamp_str}\n'
        f'Hostname: {bench_result.context.hostname}\n'
        f'Number of CPUs: {bench_result.context.num_cpus}\n'
        f'GPU Memory Capacity (GB): {math.ceil(bench_result.context.gpu_mem / (1024 * 1024 * 1024))}\n'
        f'GPU Architecture: {bench_result.context.gpu_arch}\n'
        f'\nBenchmark Results:\n'
    )

    for name, runs in bench_result.runs.items():
        file_content += (
            f'\n{separator}\n'
            f'{name}\n'
            f'\nWithout Jump Buffers (ms) | With Jump Buffer (ms) | Time Difference (ms) '
            f'| Transfer Buffer Size (MB) | Jump Buffer Size (MB) | Parallel Copy Threads | Status\n'
        )

        bench_status = 'PASS'
        for run in runs:
            time_difference = run.no_jump_buffer_run_time_ms - run.with_jump_buffer_run_time_ms
            if time_difference >= 0:
                status = 'PASS'
            else:
                bench_status = status = 'FAIL'
            transfer_buffer_size_gb = run.transfer_buffer_size / (1024 * 1024 * 1024)
            no_jump_buffer_time_and_rate = (
                f'{run.no_jump_buffer_run_time_ms:.3f} '
                f'({transfer_buffer_size_gb / (run.no_jump_buffer_run_time_ms / 1000):.2f} GB/s)'
            )
            with_jump_buffer_time_and_rate = (
                f'{run.with_jump_buffer_run_time_ms:.3f} '
                f'({transfer_buffer_size_gb / (run.with_jump_buffer_run_time_ms / 1000):.2f} GB/s)'
            )
            time_and_percent_difference = (
                f'{time_difference:+.3f} ({(time_difference / run.no_jump_buffer_run_time_ms * 100):+.2f}%)'
            )
            file_content += (
                f'{no_jump_buffer_time_and_rate:<26}'
                f'| {with_jump_buffer_time_and_rate:<22}'
                f'| {time_and_percent_difference:<21}'
                f'| {run.transfer_buffer_size / (1024 * 1024):<26.0f}'
                f'| {run.jump_buffer_size / (1024 * 1024):<22.0f}'
                f'| {run.parallel_copy_threads:<22.0f}'
                f'| {status}\n'
            )
        file_content += f'\nBenchmark Status: {bench_status}\n{separator}\n'

    with open(file_path, 'w') as file:
        file.write(file_content)

    print(f'Text file: {file_path} created')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()

    bench_result = create_benchmark_result(args.input_file)
    file_name_prefix = f'{args.tag}_'
    benchmark_result_to_json_file(bench_result, f'{file_name_prefix}bench_report.json')
    benchmark_result_to_text_file(bench_result, f'{file_name_prefix}bench_report.txt')
