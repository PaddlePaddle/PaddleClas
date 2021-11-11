// pages/main/uploadpage/upload.js
var app = getApp()
// var common = require('../../../utils/whole.js')
Page({
  /**
   * 页面的初始数据
   */
  data: {
    //系统状态栏高度
    statusbarheight:app.globalData.statusBarHeight,
    // 上传图片
    imgList: [],
    index: null,
    //表单input值，用于提交成功后清空表单
    formvalue:'',
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    
  },
  //顶部操作栏返回
  onClickBack: function(){
    wx.navigateBack({
      delta: 1
    })
  },
  //图片上传选择
  ChooseImage() {
    wx.chooseImage({
      count: 1, //默认9
      sizeType: ['original', 'compressed'], //可以指定原图还是压缩图，默认二者都有
      sourceType: ['album','camera'], //从相册选择
      success: (res) => {
        if (this.data.imgList.length != 0) {
          this.setData({
            imgList: this.data.imgList.concat(res.tempFilePaths)
          })
        } else {
          wx.setStorageSync('temp', res.tempFilePaths)
          this.setData({
            imgList: res.tempFilePaths
          })
          wx.setStorageSync('imgList', this.data.imgList)
        }
      }
    });
  },
  //图片上传预览
  ViewImage(e) {
    wx.previewImage({
      urls: this.data.imgList,
      current: e.currentTarget.dataset.url
    });
  },
  //图片上传删除
  DelImg(e) {
    this.data.imgList.splice(e.currentTarget.dataset.index, 1);
    this.setData({
      imgList: this.data.imgList
    })
  },
  //点击上传按钮操作
  formSubmit: function (e){
    var that = this
    var formData = e.detail.value
    //判断表单是否为空
    if(formData.productname&&formData.productunitprice&&this.data.imgList[0]!=null){
      wx.showModal({
        title: '上传确认',
        content: '请检查上传信息正确无误',
        cancelText: '取消',
        confirmText: '确认',
        success: res => {
          if (res.confirm) {
            wx.showLoading({
              title: '正在上传',
            })
            //判断session-key是否过期
            wx.checkSession({
              success(){
                app.globalData.isSKexpired = false
                wx.setStorageSync('isSKexpired', false)
                app.globalData.code = null
                wx.setStorageSync('code', null)
              },
              fail(){
                app.globalData.isSKexpired = true
                wx.setStorageSync('isSKexpired', true)
                wx.login({
                  success(res){
                    if(res.code){
                      app.globalData.code = res.code
                      wx.setStorageSync('code', res.code)
                    }
                  }
                })
              }
            })
            //上传表单
            wx.uploadFile({
              url: 'http://106.12.78.130/record/',
              filePath: this.data.imgList[0],
              name: 'productimage',
              formData:{
                'container_name':formData.productname,
                'container_price':formData.productunitprice,
                'sessionID':wx.getStorageSync('sessionID'),
                'isSKexpired':JSON.stringify(wx.getStorageSync('isSKexpired')),
                'code':JSON.stringify(wx.getStorageSync('code'))
              },
              method:'POST',
              header:{
                'Content-Type':'multipart/form-data'
              },
              success(res){
                var result = JSON.parse(res.data)
                wx.hideLoading()
                if(result.state=='true'){
                  app.globalData.sessionID = result.sessionID
                  wx.setStorageSync('sessionID', result.sessionID)
                  wx.showToast({
                    title: '上传成功',
                    icon:'success'
                  })
                  //上传成功，清空表单
                  that.setData({
                    formvalue:'',
                    imgList:[]
                  })
                }
                else{
                  wx.showToast({
                    title: '上传失败',
                    icon:'error'
                  })
                }
              }
            })
          }
        }
      })
    }
    else{
      wx.showModal({
        title: '信息有误',
        content: '请检查上传信息是否完整',
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