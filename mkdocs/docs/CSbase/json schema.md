https://json-schema.org/learn/getting-started-step-by-step

json schema用于 json的注释和验证，

```json
{
  "productId": 1,
  "productName": "A green door",
  "price": 12.50,
  "tags": [ "home", "green" ]
}
```

json schema可以用于描述一组json数据的结构(the structure)，约束(constraints)和数据类型(data types)

对于json的instance，可以加入约束，比如 `type` 关键字来约束一个instance：`object`, `array`, `string`, `number`, `boolean`, or `null`:


```json
{"type":"object"}
```


# shcema definition
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.com/product.schema.json",
  "title": "Product",
  "description": "A product in the catalog",
  "type": "object"
}
```

- `$schema`: 指定JSON模式标准的草稿架构遵守
- `$id`: 为schema设置URI。您可以使用此独特的URI来参考来自同一文档内部或外部JSON文档的schema元素
- `title` and `description`: 陈述 schema的意图. 这些关键字不会给数据限制
- `type`: 定义json数据的第一个限制，在下述的产品catalog中，这个关键字表示数据必须是JSON object.



# 添加 properties

properties用于验证json里的key和value，比如下面定义的properties里右两个关键字 `productId` 和 `productName`, 并且为他们的value添加了 type 限制 `integer` 或 `string` 


```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.com/product.schema.json",
  "title": "Product",
  "description": "A product from Acme's catalog",
  "type": "object",
  "properties": {
    "productId": {
      "description": "The unique identifier for a product",
      "type": "integer"
    },
    "productName": {
      "description": "Name of the product",
      "type": "string"
    }
  }
}
```

# 指定必需属性

通过添加 `"required": [ "productId", "productName", "price" ]` 表示这些key是必须的
`"exclusiveMinimum": 0` 表示最小值为0

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.com/product.schema.json",
  "title": "Product",
  "description": "A product from Acme's catalog",
  "type": "object",
  "properties": {
    "productId": {
      "description": "The unique identifier for a product",
      "type": "integer"
    },
    "productName": {
      "description": "Name of the product",
      "type": "string"
    },
    "price": {
      "description": "The price of the product",
      "type": "number",
      "exclusiveMinimum": 0
    }
  },
  "required": [ "productId", "productName", "price" ]
}
```


# 指定可选属性

添加一个tags的properties，指定类型为 `array` , `item` 表示array内的item，设置其type为 `string`，`"minItems": 1,`表示最少一个 `item`，`"uniqueItems": true` 表示 `item` 不能重复

```json
    "tags": {
       "description": "Tags for the product",
       "type": "array",
       "items": {
         "type": "string"
       },
       "minItems": 1,
       "uniqueItems": true
     }
```


# 嵌套结构 nested data structure

新建一个properties dimension，type设置为object，可以设置dimensions的properties，并且设置required

```json
  "dimensions": {
     "type": "object",
     "properties": {
       "length": {
         "type": "number"
       },
       "width": {
         "type": "number"
       },
       "height": {
         "type": "number"
       }
     },
     "required": [ "length", "width", "height" ]
 }
```


