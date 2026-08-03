[掌握hashtable，深度理解python字典的数据结构 - 酷python的文章 - 知乎](https://zhuanlan.zhihu.com/p/379440639)
[wiki hashtable](https://en.wikipedia.org/wiki/Hash_table#:~:text=In%20computing%2C%20a%20hash%20table,that%20maps%20keys%20to%20values.)

![[Pasted image 20240717000400.png]]


hashtable 的时间复杂度和空间复杂度

| Operation | **Average** | **Worst case** |     |
| --------- | ----------- | -------------- | --- |
| Search    | Θ(1)        | Θ(n)           |     |
| Insert    | Θ(1)        | Θ(n)           |     |
| Delete    | Θ(1)        | Θ(n)           |     |
| Space     | Θ(n)        | Θ(n)           |     |





# hash函数

你在网站上注册用户，密码不会明文保存，而是经过hash函数处理的密码，典型的有MD5。你在下载文件时，还会得到一个专门用来验证文件是否被串改的签名，这个签名是hash函数生成的。

hash函数是一个统称，有许多种实现算法，例如MD5、SHA-1、SHA-2、NTLM等等。

哈希函数，是一种从任何一种数据中创建小的数字“指纹”的方法，这是维基百科上的定义，换一种更容易理解的定义，*哈希函数将任意长度的数据映射到固定长度的值*。

哈希函数，有以下3个基本特性：

1. *速度必须快*，因为hash应用太广泛了，太基础了，速度慢是不可接受的
2. 映射的结果是确定的，*同一个原始输入，经hash函数处理后得到的散列值必须相同*，如果两个散列值不同，那么他们的原始输入也不相同
3. hash函数*产生固定长度的值*，这里的固定长度指的是存储散列值所占用的字节数相同

除了上面的3个基本特性，还有一个比较常见的特性，hash过程是不可逆的，这便是说，即便给了你散列值，你也无法逆向算出原始输入，在密码学里，这一点是必须保证的

## 散列碰撞
由于hash函数产生的散列值的长度是固定的，这意味着散列值的个数是有限的，而原始输入却可以是无限多个，因此，两个不同的输入，可能会得到一个相同的散列值，这个叫散列碰撞（collision），散列碰撞是无法避免的，好的hash算法只能是降低碰撞的概率，而无法杜绝。

python的内置函数hash就是一个hash函数，它可以计算任意不可变对象的hash值

```python
>>> hash('323')
-2887333350099739771
>>> hash(323)
323
>>> hash(4334.2323)
535647331039580398
>>> hash((1, 3))   
3713081631933328131
```

hash函数返回的是int类型数据，前面已经强调过，hash函数返回固定长度的值，这里的固定长度，指的是存储这个值所用的字节数，虽然这些int类型的数值大小不同，看起来长度不一致，但存储他们所用的字节数是相同的。对于自定义类，可以重写魔法方法`__hash__`来实现自己的hash函数。

对于可变对象，例如列表，是不能被hash的

```python
>>> hash([1, 2, 3])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unhashable type: 'list'
```

#  hashtable

hashtable是一种可以存储键值对的数据结果。
用顺序表来存数据  
存键值对时，通过哈希函数计算出键对应的索引，将值存到索引对应的数据区中  
获取数据时，通过哈希函数计算出键对应的索引，将该索引对应的数据取出来

遇到散列碰撞的时候就会有hash冲突，即两个不同的键会有相同的hash值，因此会指向同一个索引，此时可以是哦那个**开链法**链接冲突的元素到后面，缺点是空间占用较大。
![[Pasted image 20240717001139.png]]

也可以使用**开放寻址法**，如果哈希函数得到的位置i已经又数据了，那么就往后探查新的位置来存储这个值  线性探测：如果i有数据了，则探测`i+1`，`i+2`…以此类推，直到找到空的位置。获取值的过程  获取`dic[‘dog’]`的时候，先到索引为2的位置去获取 ，获取不到继续向后探测。删除时给节点定义一个状态，未使用，已使用，已删除


下面是python实现hashtable的一个示例，PyHashTable，并不是一个完整的实现，一些功能并没有添加，比如删除某个key，但已经将hashtable最核心的结构和操作体现出来了，写入新的key-value对，通过key获取value，操作方式和字典是一样的

```python
class PyHashTable():
    def __init__(self, datas=None):
    '''init hashtable with const length list '''
        if datas is None:
            self.length = 8
        else:
            self.length = len(datas)
	
        self.buckets = [[] for i in range(self.length)]
        self.init_buckets(datas)

    def init_buckets(self, datas):
        if datas is None:
            return
        for key, value in datas:
            self.__setitem__(key, value)

    def __getitem__(self, search_key):
	    '''get v with k'''
        hash_value = abs(hash(search_key))
        index = hash_value % self.length
        for key, value in self.buckets[index]:
            if search_key == key:
                return value

    def __setitem__(self, key, value):
        '''add k-v pair into buckets list'''
        hash_value = abs(hash(key))
        index = hash_value % self.length
        self.buckets[index].append((key, value))

datas = [('python', 90), ('java', 98), ('php', 85), ('c', 100)]
hashtable = PyHashTable(datas)
print(hashtable['c'])   # 像使用字典一样

hashtable['c++'] = 92       
print(hashtable['c++'])

print(hashtable.buckets)
```

