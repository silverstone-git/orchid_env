import subprocess
import os
import uuid

class MSBController:
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path

    def set_dataset(self, dataset_path: str):
        self.dataset_path = dataset_path

    def _get_dataset_mount_args(self):
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            return []
        
        # Mount the directory containing the dataset
        host_dir = os.path.abspath(os.path.dirname(self.dataset_path))
        filename = os.path.basename(self.dataset_path)
        guest_dir = "/data"
        # We will expect the code to read from /data/filename
        return ["-v", f"{host_dir}:{guest_dir}"]

    def run_code(self, python_code: str) -> str:
        """Runs arbitrary python code in an ephemeral MSB sandbox."""
        cmd = ["msb", "run", "--quiet"]
        mount_args = self._get_dataset_mount_args()
        cmd.extend(mount_args)
        
        # Use python image
        cmd.extend(["python", "--", "python", "-c", python_code])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return f"Error: {result.stderr.strip()}"
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            return "Error: Execution timed out"
        except Exception as e:
            return f"Error: {str(e)}"
