import subprocess
greps = ['mixed    ', 'allocated', 'idle     ', 'drained  ']
results = []


for g in greps:
    search_string = g.replace(' ', '')
    proc1 = subprocess.Popen(['slurm', 'n'], stdout=subprocess.PIPE)
    proc2 = subprocess.Popen(['grep', '-c', search_string], stdin=proc1.stdout,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc1.stdout.close() 
    out, err = proc2.communicate()
    str_out = out.decode()
    results.append(int(str_out.rstrip(str_out[-1])))

print(results)
total = sum(results)

for idx in range(len(greps)):
    print(str(greps[idx]) + ': \t' + str(results[idx]) + '\t' + str(round((float(results[idx]) / total )*100)) + ' %')
