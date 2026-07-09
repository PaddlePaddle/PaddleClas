"""
下载官方 Kinetics-400 类别列表
"""
import urllib.request
import ssl

url = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

print("Downloading K400 class list...")
response = urllib.request.urlopen(url, timeout=60, context=ssl_context)
content = response.read().decode('utf-8')

classes = [line.strip() for line in content.strip().split('\n') if line.strip()]

print(f"Found {len(classes)} classes")
print("First 10:", classes[:10])
print("Last 10:", classes[-10:])

with open("k400_classes.txt", "w") as f:
    for c in classes:
        f.write(f'"{c}",\n')

print("\nSaved to k400_classes.txt")

print("\nPython list format:")
print("K400_CLASSES = [")
for c in classes:
    print(f'    "{c}",')
print("]")
