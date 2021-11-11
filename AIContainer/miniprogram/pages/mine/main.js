// miniprogram/pages/mine/main.js
var app = getApp()
Page({

  /**
   * 页面的初始数据
   */
  data: {
    //用户信息
    userInfo:{},
    hasUserInfo:false,
    canIUseGetUserProfile:false,
    //商品信息
    productnumber:null,
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    //判断是否使用getUserProfile
    if(wx.getUserProfile){
      this.setData({
        canIUseGetUserProfile:true,
      })
    }
    //判断是否有缓存的用户信息
    if(wx.getStorageSync('userInfo')){
      this.setData({
        hasUserInfo:true,
      })
    }
    //设置userInfo
    this.setData({
      userInfo:wx.getStorageSync('userInfo'),
    })
  },
  //获取用户信息
  getUserProfile(e) {
    // 使用wx.getUserProfile获取用户信息，每次通过该接口获取用户个人信息均需用户确认
    wx.getUserProfile({
      desc: '用于完善用户信息', 
      // lang:'zh_CN',
      success: (res) => {
        app.globalData.userInfo=res.userInfo,
        wx.setStorageSync('userInfo', app.globalData.userInfo)
        this.setData({
          userInfo:app.globalData.userInfo,
          hasUserInfo: true
        })
        wx.login({
          success(res){
            if(res.code){
              console.log(app.globalData.userInfo)
              wx.request({
                url: 'http://106.12.78.130/login_in/',
                data:{
                  'code':res.code,
                  'userinfo':JSON.stringify(app.globalData.userInfo)
                },
                method: 'POST',
                header: {
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                success: res=>{
                  app.globalData.sessionID = res.data.sessionID
                  wx.setStorageSync('sessionID', app.globalData.sessionID)
                  console.log(res)
                }
              })
            }
          }
        })
      }
    })
    // wx.login({
    //   success(res){
    //     if(res.code){
    //       console.log(app.globalData.userInfo)
    //       wx.request({
    //         url: 'http://127.0.0.1:8000/login_in/',
    //         data:{
    //           'code':res.code,
    //           'userinfo':app.globalData.userInfo
    //         },
    //         method: 'POST',
    //         header: {
    //             "Content-Type": "application/x-www-form-urlencoded"
    //         },
    //         success: res=>{
    //           console.log(res)
    //         }
    //       })
    //     }
    //   }
    // })

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