// pages/main/recognitionpage/recognition.js
var app = getApp()
Page({
  /**
   * 页面的初始数据
   */
  data: {
    //是否显示相机
    isCamera:true,
    //拍摄的照片
    photo:'',
    //是否显示识别结果
    isResult:false,
    //识别结果图片
    resultimg:'',
    //识别结果
    result:[],
    //总价
    price_all:'',
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
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
  },
  //拍照按钮，捕获相片地址
  takePhoto() {
    var ctx = wx.createCameraContext()
    ctx.takePhoto({
      quality: 'high',
      success: (res) => {
        this.setData({
          photo: res.tempImagePath,
          isCamera:false,
        })
        console.log(res.tempImagePath)
      }
    })
  },
  error(e) {
    console.log(e.detail)
  },
  //打开相册，从相册选图
  openalbum:function(){
    wx.chooseImage({
      count: 1,
      sizeType: ['original', 'compressed'],
      sourceType: ['album'],
      success:res=>{
        this.setData({
          photo: res.tempFilePaths[0],
          isCamera:false,
        })
      }
    })
  },
  //预览相片，返回相机按钮
  gobackCamera:function(){
    this.setData({
      photo:'',
      isCamera:true
    })
  },
  //上传相片按钮
  photoUpload:function(){
    wx.showLoading({
      title: '识别中',
    })
    //获取图片文件内容，转换成base64格式上传
    wx.getFileSystemManager().readFile({
      filePath: this.data.photo,
      encoding: "base64",
      success: res => {
        var image_base64 = res.data
          wx.request({
            url: 'http://106.12.78.130/reference/',
            method:'POST',
            header: {"content-type": "application/x-www-form-urlencoded"},
            data:{
              'sessionID':wx.getStorageSync('sessionID'),
              'isSKexpired':JSON.stringify(wx.getStorageSync('isSKexpired')),
              'code':JSON.stringify(wx.getStorageSync('code')),
              'picture':image_base64
            },
            success:res=>{
              wx.hideLoading()
              console.log(res.data.container.length)
              if(res.data.container=="Please connect root to upload container's name and it's price!\n"){
                this.setData({
                  isResult:true,
                })
                wx.showModal({
                  title: '识别错误',
                  content: '存在未知商品，',
                  showCancel:false,
                  confirmText: '确认',
                })
              }
              else{
                for(var i=0;i<=res.data.container.length/2;i=i+2){
                  var temp = []
                  temp.push(res.data.container[i])
                  temp.push(res.data.container[i+1])
                  this.data.result.push(temp)
                }
                this.setData({
                  isResult:true,
                  result:this.data.result,
                  resultimg:res.data.picture_test,
                  price_all:res.data.price_all,
                })
              }
            }
          })
      },
    })
  },
  //下单付款按钮
  onClickBuy:function(){
    
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