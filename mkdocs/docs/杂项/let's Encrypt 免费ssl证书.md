[借助 NGINX 在 Let’s Encrypt 中使用免费 SSL/TLS 证书 - NGINX开源社区的文章 - 知乎](https://zhuanlan.zhihu.com/p/678640013)

https://certbot.eff.org/instructions?ws=nginx&os=ubuntufocal&tab=standard

[certbot签发和续费泛域名SSL证书（通过DNS TXT验证域名有效性）](https://blog.csdn.net/cljdsc/article/details/133461361)

注意，ubuntu apt源的 certbot不适配ubuntu 20.04，会报错 [AttributeError: module ‘acme.challenges’ has no attribute ‘TLSSNI01’](https://community.letsencrypt.org/t/ubuntu-20-04-any-tips-attributeerror-module-acme-challenges-has-no-attribute-tlssni01/115831) （[https://github.com/certbot/certbot/issues/7951](https://github.com/certbot/certbot/issues/7951)）

因此需要 snap安装，然后手动连接

```bash
sudo snap install --classic certbot
sudo ln -s /snap/bin/certbot /usr/bin/certbot
```

官方的泛域名证书申请并配置到nginx

```bash
sudo certbot --nginx
```

但是这样是直接使用http验证，因为是内网机器，我们改为用DNS验证

```bash
sudo certbot certonly --manual --preferred-challenges dns
```

然后会提示你部署DNS txt解析

```bash
please deploy a DNS TXT record under the name:

_acme-challenge.952712138.xyz.

with the following value:

ogu.........................

before continue, verify the TXT record has been deplyed.it take minutes.

press enter to continue.

```

然后我们在托管域名的cloudflare上添加DNS记录，类型为 `TXT`，Name为 `_acme-challenge` 值为上述值。然后等待几分钟DNS解析。

使用下面命令查询解析;

```bash
nslookup -type=txt _acme-challenge.example.com 
或
dig -t txt _acme-challenge.example.com
```

比如 example.com

```
nslookup -type=txt _acme-challenge.example.com 
```

然后press enter continue成功申请证书，存在 `/etc/letsencrypt/live/example.com`

```
successfully recieved certificate
Certificate is save at: /etc/letsencrypt/live/example.com /fullchain.pem
Key is saved at: /etc/letsencrypt/live/example.com /privkey.pem
```

然后修改一下目录的权限吗，方便其他应用使用证书。

```bash
sudo chmod 775 -R /etc/letsencrypt
```

在nginx配置文件里配置证书即可。

设置自动续期证书，将下列命令写入crond（linux定时任务）（疑似没有生效）

```bash
0 3 */7 * * /bin/certbot renew --renew-hook "/where/your/nginx -s reload" 
```
