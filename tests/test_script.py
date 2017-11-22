import subprocess
import sys
from datetime import datetime
import os.path

n_iterations = 11 # we want 10 iterations and we will ignore the first (might be affected by non-cached data)

if len(sys.argv) < 2:
	print("provide a path to a process to execute")
	exit()

logfile_dir = "./"
if len(sys.argv) > 2:
	logfile_dir = sys.argv[2] + "/"

process_path = sys.argv[1]
_, process_name = os.path.split(sys.argv[1])
logfile_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_" + process_name + ".csv"

print("running " + str(n_iterations) + " instances of " + process_name + " and writing performance measures to " + logfile_dir + logfile_name)

with open(logfile_dir + logfile_name, "w") as f:
	print("num_elements, num_cycles, time_compute, time_copy", file=f)
	for i in range(n_iterations):
		result = subprocess.run([process_path], stdout=subprocess.PIPE).stdout.decode('utf-8')
		print(", ".join(s for s in result.split() if s.isdigit()), file=f) # assuming that only the last lines prints the three performance numbers
		print(".", end='', flush=True)

print("")
