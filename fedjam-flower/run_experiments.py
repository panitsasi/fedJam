import subprocess
import time
import os
import toml
import shutil
import signal
from params.params_dense_net_all_channels import param_grid

# CONFIGURATION
BASE_TOML = "pyproject.toml"
BACKUP_TOML = "pyproject.backup.toml"
GPU_POOL = ["0"]  
MAX_PARALLEL = len(GPU_POOL)
DELAY_BETWEEN_LAUNCHES = 20  # seconds

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

running_jobs = []
available_gpus = set(GPU_POOL)

def launch_experiment(i, params, gpu_id):
    print(f"[INFO] Launching experiment {i} on GPU {gpu_id}")

    # Load and update config
    config = toml.load(BACKUP_TOML)
    for (section, param), value in params.items():
        sections = section.split('.')
        d = config['tool']['flwr']
        for s in sections:
            if s not in d:
                d[s] = {}
            d = d[s]
        d[param] = value

    # Write updated config
    with open(BASE_TOML, 'w') as f:
        toml.dump(config, f)

    log_file = f"logs/experiment_{i}.log"
    log_f = open(log_file, "w")

    cmd = ["flwr", "run"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    process = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)

    return (process, gpu_id, i, log_f)

def run_experiments_with_gpu_queue(param_grid):
    if not os.path.exists(BACKUP_TOML):
        shutil.copy(BASE_TOML, BACKUP_TOML)

    job_queue = list(enumerate(param_grid))

    try:
        while job_queue or running_jobs:
            # Watch for STOP file
            if os.path.exists("STOP"):
                print("[INFO] STOP file detected. Terminating all running jobs.")
                break

            # Launch new jobs if GPUs available
            while available_gpus and job_queue:
                i, params = job_queue.pop(0)
                assigned_gpu = sorted(available_gpus)[0]
                available_gpus.remove(assigned_gpu)

                print(f"[INFO] Waiting {DELAY_BETWEEN_LAUNCHES}s before launching experiment {i}...")
                time.sleep(DELAY_BETWEEN_LAUNCHES)

                job = launch_experiment(i, params, assigned_gpu)
                running_jobs.append(job)

            # Poll running jobs
            new_running_jobs = []
            for proc, gpu_id, exp_index, log_f in running_jobs:
                if proc.poll() is None:
                    new_running_jobs.append((proc, gpu_id, exp_index, log_f))
                else:
                    print(f"[INFO] Experiment {exp_index} on GPU {gpu_id} completed.")
                    available_gpus.add(gpu_id)
                    with open(f"logs/experiment_{exp_index}.done", "w") as done_f:
                        done_f.write("done\n")
                    log_f.close()
            running_jobs[:] = new_running_jobs

            time.sleep(10)

    except KeyboardInterrupt:
        print("\n[WARNING] KeyboardInterrupt detected. Cleaning up...")

    finally:
        # Kill any running jobs
        for proc, gpu_id, exp_index, log_f in running_jobs:
            print(f"[INFO] Terminating experiment {exp_index} on GPU {gpu_id}")
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            log_f.close()

        # Restore original config
        if os.path.exists(BACKUP_TOML):
            shutil.move(BACKUP_TOML, BASE_TOML)
            print("[INFO] Restored original pyproject.toml")

        print("[INFO] All experiments handled. Exiting.")

if __name__ == "__main__":
    run_experiments_with_gpu_queue(param_grid)
