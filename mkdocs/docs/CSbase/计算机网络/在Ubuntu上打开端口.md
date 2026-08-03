
# 查看端口状态

查看当前的网络连接状态，包括本机地址、远程地址、协议类型、连接状态以及所使用的端口。
```bash
netstat -an
```


查看当前系统的防火墙策略，包括所有的规则集合以及针对每个规则集合所采取的不同策略。如果该命令输出结果为空，则表示防火墙没有任何规则。
```bash
sudo iptables -L -n
```

# 打开端口

## 使用ufw

ufw是Ubuntu的默认防火墙，可以使用以下命令查看防火墙的状态：


```bash
sudo ufw status verbose
```


如果防火墙状态是`inactive`，则表示没有启动防火墙。如果防火墙状态是`active`，则表示已经启动防火墙。

可以使用以下命令打开一个端口：


```bash
sudo ufw allow [端口号]/[协议]
```


其中，端口号是需要打开的端口号，协议是需要打开的协议类型，如`TCP`或`UDP`。例如，打开端口`22`（SSH）可以使用以下命令：


```bash
sudo ufw allow 22/tcp
```


## 使用iptables

`iptable`是Linux系统的默认防火墙，它可以控制内核模块来实现网络访问的控制。可以使用以下命令打开一个端口：


```bash
sudo iptables -A INPUT -p [协议] --dport [端口号] -j ACCEPT
```


其中，协议是需要打开的协议类型，如TCP或UDP，端口号是需要打开的端口号。例如，打开端口22（SSH）可以使用以下命令：


```bash
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
```


# 保存防火墙配置

无论是使用ufw还是iptables打开端口，都需要保存防火墙配置，以保证下次启动系统时配置不会被清空。

使用ufw的保存命令如下：


```bash
sudo ufw enable
```


使用iptables的保存命令如下：


```bash
sudo service iptables save
```


四、测试端口连接

完成端口打开之后，需要测试连接是否成功。可以使用以下命令连接到一个打开的端口：


```bash
telnet [主机名] [端口号]
```


其中，主机名是远程主机的IP地址或域名，端口号是需要连接的端口号。例如，连接到本地的端口22可以使用以下命令：


```bash
telnet localhost 22
```



