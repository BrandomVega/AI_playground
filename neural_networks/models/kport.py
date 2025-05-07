import os
import sys
import subprocess

def kill_process_by_port(port):
    result = subprocess.run(f"lsof -t -i :{port}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode == 0:
        pids = result.stdout.decode().strip().split('\n')
        for pid in pids:
            try:
                os.kill(int(pid), 9)  
                print(f"Proceso con PID {pid} terminado")
            except Exception as e:
                print(f"No se pudo terminar el proceso {pid}: {e}")
    else:
        print(f"No se encontraron procesos usando el puerto {port}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python3 script.py <puerto>")
        sys.exit(1)
    
    puerto = sys.argv[1]
    try:
        puerto = int(puerto)
    except ValueError:
        print("Invalid port")
        sys.exit(1)
    
    kill_process_by_port(puerto)
