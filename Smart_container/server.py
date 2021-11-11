import socket

# 创建一个tcp socket ip 地址版本为Ip4
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 绑定IP和端口号
sock.bind(('106.12.78.130',8020))
#最大限度接收数
sock.listen(128)

print('服务器开启再：',8080)
# 取出新的socket  addr是IP，而端口是随机生成的底层做了
# accept()处理的连接都是三次握手的连接
new_cli,addr = sock.accept()
# 打印一下IP类型是个元祖
print('来自一个新的连接', addr,type(addr))
# 接收数据的大小
data = new_cli.recv(1024)
print(data)
print(new_cli)
new_cli.close()
sock.close()

