// miniprogram/pages/main/main.js
var app = getApp()
Page({
  /**
   * 页面的初始数据
   */
  data: {
    //搜索框
    searchtarget:'',
    pictures:[],
    elements:[{
      title:'上传',
      name : 'upload',
      icon:'pullup',
      color:'deepskyblue'
    },
    {
      title : '修改',
      name : 'revise',
      icon:'moreandroid',
      color:'blue'
    },
  ],
  element:[
    {
      title : '识别',
      name : 'recognition',
      icon:'explorefill',
      color:'cyan'
    }
  ]

  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    //设置轮播图data的pictures图片路径
    this.setData({
      searchtarget:''
    })
  },
  //搜索框文本内容显示
  inputBind: function(event) {
    this.setData({
        searchtarget: event.detail.value
    })
  },
  //清空搜索框
  onClickCleanSearch(){
    this.setData({
      searchtarget:''
    })
  },
  //搜索框检索请求
  getSearch(){
    var that = this
    if(this.data.searchtarget){
      wx.request({
        url: 'http://106.12.78.130/find/',
        data:{
          'sessionID':wx.getStorageSync('sessionID'),
          'isSKexpired':JSON.stringify(wx.getStorageSync('isSKexpired')),
          'code':JSON.stringify(wx.getStorageSync('code')),
          'searchtarget':this.data.searchtarget
        },
        method: 'POST',
        header: {
            "Content-Type": "application/x-www-form-urlencoded"
        },
        success:res=>{
          console.log(res.data)
          if(res.data.container_all.length>0){
            this.setData({
              list:res.data.container_all
            })
            wx.setStorageSync('product_all', res.data.container_all)
            wx.navigateTo({
              url: '/pages/main/revisepage/revise?data='+JSON.stringify(res.data.container_all)+'&&searchtarget='+JSON.stringify(this.data.searchtarget),
            })
          }
          else{
            wx.showToast({
              title: '商品不存在',
              icon:'error'
            })
          }
        }
      })
    }
    else{
      wx.showModal({
        title: '信息有误',
        content: '请输入所要查找的商品名称',
        showCancel:false,
        confirmText: '确认',
      })
    }
  },
  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function () {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {

  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide: function () {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload: function () {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function () {

  }
})