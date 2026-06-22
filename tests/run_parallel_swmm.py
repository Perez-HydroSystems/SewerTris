import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_one_model(inp_path, worker_script, timeout_seconds=None):
    try:
        result = subprocess.run(
            [sys.executable, str(worker_script), str(inp_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"Timed out after {timeout_seconds} seconds while running {inp_path}"
        ) from exc

    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(details or f"Worker exited with status {result.returncode}")

    return {
        "inp_path": inp_path,
        "flow_path": inp_path.with_name(f"{inp_path.stem}_flows.nc"),
        "stdout": result.stdout,
    }

def run_models_parallel(
    output_folder,
    n_parallel_models=2,
    inp_pattern="sewer_model_*.inp",
    inp_files=None,
    timeout_seconds=3600,
):
    output_folder = Path(output_folder)
    worker_script = Path(__file__).parent / "swmm_worker.py"

    if not output_folder.exists():
        raise FileNotFoundError(f"Output folder does not exist: {output_folder}")

    if not worker_script.exists():
        raise FileNotFoundError(f"Worker script does not exist: {worker_script}")

    if inp_files is None:
        inp_files = sorted(output_folder.glob(inp_pattern))
    else:
        inp_files = sorted(Path(path) for path in inp_files)

    if not inp_files:
        raise FileNotFoundError(f"No INP files found in {output_folder}")

    requested_workers = max(1, int(n_parallel_models))
    max_workers = min(requested_workers, len(inp_files), os.cpu_count() or 1)

    print(f"Found {len(inp_files)} models")
    print(f"Running {max_workers} models in parallel")
    if timeout_seconds is not None:
        print(f"Worker timeout: {timeout_seconds} seconds")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_one_model, inp, worker_script, timeout_seconds): inp
            for inp in inp_files
        }

        for future in as_completed(futures):
            inp = futures[future]

            try:
                result = future.result()
                results.append(result)
                print(f"Finished: {result['inp_path'].name}")
                print(result["stdout"])

            except Exception as e:
                print(f"Failed: {inp.name}")
                print(e)
                raise

    return sorted(results, key=lambda item: item["inp_path"].name)
