[ddns-go](https://github.com/jeessy2/ddns-go)可以自动获得你的公网 IPv4 或 IPv6 地址，并解析到对应的域名服务。

# 安装
参考官方文档




# 多个ipv6的问题
https://ruohai.wang/202312/ddns-go-choose-the-last-ipv6/

具体表现就是：

- 出现了2~3个公网ipv6地址
- 只有最新的那个ipv6地址才是有效的
- 旧的ipv6地址会一直保留直到它的valid_lft有效生命周期结束

移动宽带定期（maybe不定期）pppoe重播，导致不断分配新的公网ipv6地址，但旧的公网ipv6地址并没有被主动弃用/注销，所以出现了多个公网ipv6地址共存、但只有最新的那个公网ipv6地址有效的情况。

知道原因以后，找解决方案就很简单，只需要动态获取`最新的那个公网ipv6地址`即可。

ddns-go获取ipv6地址有三种方式：

1. 根据接口
2. 根据网卡
3. 根据命令

第一个方式在出现多个公网ipv6地址的时候会gg，表现就是访问不到接口，导致无法获取ipv6地址。第二个方式只支持最简单的用`@1@2@3`来指定使用第1、2、3个ipv6地址，如果最多只出现2个ipv6地址的话可以用`@2`，但如果出现3个ipv6地址，就搞不定了。

所以只有第三个方式了，用命令在动态选择最新的那个ip6v地址。

查看了多个issue以后，还是刚刚提到的这个帖子：[#872 希望ipv6地址获取可以根据valid_lft或preferred_lft排序选择](https://github.com/jeessy2/ddns-go/issues/872)，有人提供了自己写的命令。

根据issue，可以使用命令获取

```bash
ip addr show|grep -v deprecated|grep -A1 'inet6 [^f:]'|grep -v ^--|sed -nr ':a;N;s#^ +inet6 ([a-f0-9:]+)/.+? scope global .*? valid_lft ([0-9]+sec) .*#\2 \1#p;Ta'|sort -nr|head -n1|cut -d' ' -f2
```
