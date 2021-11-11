## WeUI组件库简介

[![](https://img.shields.io/npm/v/weui-miniprogram.svg?style=flat-square)](https://www.npmjs.com/package/weui-miniprogram)
[![](https://img.shields.io/npm/dw/weui-miniprogram?style=flat-square)](https://www.npmjs.com/package/weui-miniprogram)
[![](https://img.shields.io/travis/wechat-miniprogram/weui-miniprogram.svg?style=flat-square)](https://github.com/wechat-miniprogram/weui-miniprogram)
[![](https://img.shields.io/github/license/wechat-miniprogram/weui-miniprogram.svg?style=flat-square)](https://github.com/wechat-miniprogram/weui-miniprogram)
[![](https://img.shields.io/codecov/c/github/wechat-miniprogram/weui-miniprogram.svg?style=flat-square)](https://github.com/wechat-miniprogram/weui-miniprogram)

这是一套基于样式库[weui-wxss](https://github.com/Tencent/weui-wxss/)开发的小程序扩展组件库，同微信原生视觉体验一致的扩展组件库，由微信官方设计团队和小程序团队为微信小程序量身设计，令用户的使用感知更加统一。

## 如何使用
详细使用参考[文档](https://wechat-miniprogram.github.io/weui/docs/)

## 开发
1. 初始化
```
npm run init
```

2. 执行命令：

```
npm run dev
```

3. 构建组件库：

```
npm run build
```

## 适配 DarkMode

在根结点（或组件的外层结点）增加属性 `data-weui-theme="dark"`，即把 WeUI 组件切换到 DarkMode 的表现，如:

```html
<view data-weui-theme="dark">
    ...
</view>
```
