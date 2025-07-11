"""
Utility functions for remake.py - contains spinner, result parsing, and printing functions
"""
import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor
import subprocess


class Spinner:
    """Simple spinner for indicating progress with elapsed time"""
    def __init__(self, message="Working"):
        self.message = message
        self.spinner_chars = "|/-\\"
        self.running = False
        self.thread = None
        self.start_time = None
        
    def spin(self):
        idx = 0
        while self.running:
            char = self.spinner_chars[idx % len(self.spinner_chars)]
            elapsed = time.time() - self.start_time
            elapsed_str = self._format_time(elapsed)
            sys.stdout.write(f"\r{self.message}... {char} [{elapsed_str}]")
            sys.stdout.flush()
            time.sleep(0.1)
            idx += 1
    
    def _format_time(self, seconds):
        """Format elapsed time as MM:SS or HH:MM:SS if over an hour"""
        if seconds < 3600:  # Less than 1 hour
            mins, secs = divmod(int(seconds), 60)
            return f"{mins:02d}:{secs:02d}"
        else:  # 1 hour or more
            hours, remainder = divmod(int(seconds), 3600)
            mins, secs = divmod(remainder, 60)
            return f"{hours:01d}:{mins:02d}:{secs:02d}"
    
    def start(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self.spin)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.start_time:
            elapsed = time.time() - self.start_time
            elapsed_str = self._format_time(elapsed)
            sys.stdout.write(f"\r{self.message}... Done! [{elapsed_str}]\n")
        else:
            sys.stdout.write(f"\r{self.message}... Done!\n")
        sys.stdout.flush()


def run_command(cmd_info):
    """Execute a command and return the result"""
    year, cmd, verbose = cmd_info
    print(f"Starting command for year {year}")
    try:
        if verbose:
            # Run with real-time output
            result = subprocess.run(cmd, shell=True, text=True)
            return year, result.returncode, "", ""
        else:
            # Run with captured output
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            print(f"Completed command for year {year} (exit code: {result.returncode})")
            if result.returncode != 0:
                print(f"Error in year {year}: {result.stderr}")
            return year, result.returncode, result.stdout, result.stderr
    except Exception as e:
        print(f"Exception in year {year}: {e}")
        return year, -1, "", str(e)


def run_command_with_spinner(cmd_info):
    """Execute a command with spinner for non-verbose mode"""
    year, cmd, verbose = cmd_info
    
    if verbose:
        # Run with real-time output, no spinner
        print(f"Starting command for year {year}")
        try:
            result = subprocess.run(cmd, shell=True, text=True)
            return year, result.returncode, "", ""
        except Exception as e:
            print(f"Exception in year {year}: {e}")
            return year, -1, "", str(e)
    else:
        # Run with spinner and captured output
        spinner = Spinner(f"Processing {year}")
        spinner.start()
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            spinner.stop()
            if result.returncode != 0:
                print(f"Error in year {year}: {result.stderr}")
            return year, result.returncode, result.stdout, result.stderr
        except Exception as e:
            spinner.stop()
            print(f"Exception in year {year}: {e}")
            return year, -1, "", str(e)


def execute_commands(commands, parallel=True, use_spinner=False):
    """Execute a list of commands either in parallel or sequentially"""
    if parallel:
        # Limit parallelism to 6 cores maximum
        max_workers = min(6, len(commands))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if use_spinner:
                return list(executor.map(run_command_with_spinner, commands))
            else:
                return list(executor.map(run_command, commands))
    else:
        results = []
        for cmd_info in commands:
            if use_spinner:
                result = run_command_with_spinner(cmd_info)
            else:
                result = run_command(cmd_info)
            results.append(result)
        return results


def parse_fit_results(stdout):
    """Parse fit results from combine output"""
    signal_strengths = {}
    uncertainties = {}
    fit_status = None
    lines = stdout.split('\n')
    
    for line in lines:
        # Look for fit status information
        if 'fit status' in line.lower() or 'minimization finished successfully' in line.lower():
            fit_status = line.strip()
        elif 'converged' in line.lower() or 'covariance matrix' in line.lower():
            if not fit_status:  # Only set if we haven't found a more specific status
                fit_status = line.strip()
        
        # Look for lines containing signal strength results
        # Format typically: "r_p :  +1.234  +0.567/-0.432  (68% CL)"
        if ':' in line and any(param in line for param in ['r_p', 'r_b', 'r_q']):
            parts = line.strip().split(':')
            if len(parts) >= 2:
                param_name = parts[0].strip()
                value_part = parts[1].strip()
                
                try:
                    # Extract central value (first number after colon)
                    value_tokens = value_part.split()
                    central_value = value_tokens[0].replace('+', '')
                    signal_strengths[param_name] = central_value
                    
                    # Extract uncertainties if present
                    # Look for format like "+0.567/-0.432" or "±0.567"
                    for token in value_tokens[1:]:
                        if '+' in token and '/' in token:
                            # Asymmetric uncertainties: +0.567/-0.432
                            if token.startswith('+'):
                                uncertainties[param_name] = token
                            break
                        elif '±' in token or '+/-' in token:
                            # Symmetric uncertainties
                            uncertainties[param_name] = token
                            break
                        elif token.startswith('(') and 'CL' in token:
                            # Skip confidence level info
                            break
                except (IndexError, ValueError):
                    pass
    
    return signal_strengths, uncertainties, fit_status


def parse_limit_results(stdout):
    """Parse limit results from combine output"""
    limits = {}
    lines = stdout.split('\n')
    
    for line in lines:
        # Look for lines containing limit results
        # Format typically: "Expected 50.0%: r < 1.234"
        if 'Expected' in line and '%:' in line and '<' in line:
            parts = line.strip().split(':')
            if len(parts) >= 2:
                quantile = parts[0].strip()
                limit_part = parts[1].strip()
                
                try:
                    # Extract limit value
                    limit_value = limit_part.split('<')[1].strip()
                    limits[quantile] = limit_value
                except (IndexError, ValueError):
                    pass
        # Also look for observed limits
        elif 'Observed Limit:' in line:
            try:
                observed_limit = line.split(':')[1].strip()
                limits['Observed'] = observed_limit
            except (IndexError, ValueError):
                pass
    
    return limits


def print_execution_summary(results, title="Execution Summary"):
    """Print a summary of command execution results"""
    print(f"\n=== {title} ===")
    for year, exit_code, stdout, stderr in results:
        status = "SUCCESS" if exit_code == 0 else "FAILED"
        print(f"Year {year}: {status} (exit code: {exit_code})")
        if exit_code != 0 and stderr:
            print(f"  Error: {stderr.strip()}")
    print("=" * (len(title) + 8) + "\n")


def print_fit_summary(results):
    """Print detailed fit results summary"""
    print("\n=== Fit Summary ===")
    for year, exit_code, stdout, stderr in results:
        status = "SUCCESS" if exit_code == 0 else "FAILED"
        print(f"Year {year}: {status} (exit code: {exit_code})")
        
        if exit_code == 0 and stdout:
            signal_strengths, uncertainties, fit_status = parse_fit_results(stdout)
            
            # Display results
            if fit_status:
                print(f"  Fit status: {fit_status}")
            
            if signal_strengths:
                print(f"  Signal strengths:")
                for param, value in signal_strengths.items():
                    uncertainty_str = ""
                    if param in uncertainties:
                        uncertainty_str = f" {uncertainties[param]}"
                    print(f"    {param}: {value}{uncertainty_str}")
            else:
                # If parsing fails, show relevant lines from output
                lines = stdout.split('\n')
                relevant_lines = [line for line in lines if any(keyword in line.lower() for keyword in ['r_p', 'r_b', 'r_q', 'best fit', 'fit status', 'converged', 'covariance'])]
                if relevant_lines:
                    print(f"  Fit results:")
                    for line in relevant_lines[:8]:  # Show first 8 relevant lines
                        print(f"    {line.strip()}")
        
        if exit_code != 0 and stderr:
            print(f"  Error: {stderr.strip()}")
    print("===================\n")


def print_limit_summary(results):
    """Print detailed limit results summary"""
    print("\n=== Limit Summary ===")
    for year, exit_code, stdout, stderr in results:
        status = "SUCCESS" if exit_code == 0 else "FAILED"
        print(f"Year {year}: {status} (exit code: {exit_code})")
        
        if exit_code == 0 and stdout:
            limits = parse_limit_results(stdout)
            
            # Display results
            if limits:
                print(f"  Limits:")
                # Show in standard order if available
                limit_order = ['Expected  2.5%', 'Expected 16.0%', 'Expected 50.0%', 'Expected 84.0%', 'Expected 97.5%', 'Observed']
                for key in limit_order:
                    if key in limits:
                        print(f"    {key}: r < {limits[key]}")
                # Show any other limits not in standard order
                for key, value in limits.items():
                    if key not in limit_order:
                        print(f"    {key}: r < {value}")
            else:
                # If parsing fails, show relevant lines from output
                lines = stdout.split('\n')
                relevant_lines = [line for line in lines if any(keyword in line.lower() for keyword in ['expected', 'observed', 'limit', 'r <'])]
                if relevant_lines:
                    print(f"  Limit results:")
                    for line in relevant_lines[:10]:  # Show first 10 relevant lines
                        print(f"    {line.strip()}")
        
        if exit_code != 0 and stderr:
            print(f"  Error: {stderr.strip()}")
    print("===================\n")
