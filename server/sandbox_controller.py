import subprocess
import os
import tempfile
import sys

class LocalController:
    """
    A controller that executes code directly on the local machine WITHOUT sandboxing.
    WARNING: This is highly insecure and should only be used for trusted code or 
    local development/debugging where MSB/KVM is unavailable.
    """
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path

    def set_dataset(self, dataset_path: str):
        self.dataset_path = dataset_path

    def run_code(self, python_code: str) -> str:
        """Runs arbitrary python code locally."""
        # Create a temporary script file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            # We need to simulate the environment: chunk_data or /data/path
            # In the local version, we can just map the path or read the file.
            # But the orchestrator expects /data/... inside the code.
            # Let's mock that by creating a symlink or just replacing the path in the code.
            
            # The orchestrator code usually looks like:
            # with open('/data/filename', 'r') as f:
            
            # We will replace /data/ with the actual host directory in the code for execution.
            if self.dataset_path:
                host_dir = os.path.abspath(os.path.dirname(self.dataset_path))
                python_code = python_code.replace('/data/', f"{host_dir}/")
            
            tmp.write(python_code)
            tmp_path = tmp.name

        try:
            # Run the script using the current python interpreter
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return f"Error: {result.stderr.strip()}"
            return result.stdout.strip()
        
        except subprocess.TimeoutExpired:
            return "Error: Execution timed out"
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
