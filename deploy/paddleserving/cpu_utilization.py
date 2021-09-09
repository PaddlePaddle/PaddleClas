import psutil
cpu_utilization=psutil.cpu_percent(1,False)
print('CPU_UTILIZATION:', cpu_utilization)

