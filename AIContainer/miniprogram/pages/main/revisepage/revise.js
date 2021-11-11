// pages/main/revisepage/revise.js
var app = getApp()
Page({
  /**
   * 页面的初始数据
   */
  data: {
    //系统状态栏高度
    statusbarheight:app.globalData.statusBarHeight,
    //搜索框
    searchtarget:'',
    //回到顶部按钮
    showBackTop: false, 
    //顶部距离
    topPosition: 0, 
    //商品列表
    list:[],
    //详情页数据
    modalName:null,
    //修改图片参数
    isimageRevised:false,
    // 上传图片
    imgList: [],
    index: null,
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
    //首页搜索框传值
    if(JSON.stringify(options)!='{}'){
      var getfinddata = JSON.parse(options.data)
      var getfindsearchtarget = JSON.parse(options.searchtarget)
      this.setData({
        list:getfinddata,
        searchtarget:getfindsearchtarget,
      })
    }
    else{
      wx.showLoading({
        title: '加载中',
      })
      //请求商品列表数据
      wx.request({
        url: 'http://106.12.78.130/search/',
        data:{
          'sessionID':wx.getStorageSync('sessionID'),
          'isSKexpired':JSON.stringify(wx.getStorageSync('isSKexpired')),
          'code':JSON.stringify(wx.getStorageSync('code')),
        },
        method: 'POST',
        header: {
            "Content-Type": "application/x-www-form-urlencoded"
        },
        success:res=>{
          var that = this
          setTimeout(function(){
            that.setData({
              list:res.data.container_all
            }),
            wx.setStorageSync('product_all', res.data.container_all)
          }, 2000);
          setTimeout(function(){
            wx.hideLoading()
          },3000)
          console.log(res)
          console.log(res.data.container_all[100])
        }
      })
    }
  },
  //搜索框文本内容显示
  inputBind: function(event) {
    this.setData({
        searchtarget: event.detail.value
    })
  },
  //清空搜索框
  onClickCleanSearch(){
    //判断输入框是否为空
    if(this.data.searchtarget){
      wx.showLoading({
        title: '加载中',
      })
      //请求商品列表数据
      wx.request({
        url: 'http://106.12.78.130/search/',
        data:{
          'sessionID':wx.getStorageSync('sessionID'),
          'isSKexpired':JSON.stringify(wx.getStorageSync('isSKexpired')),
          'code':JSON.stringify(wx.getStorageSync('code')),
        },
        method: 'POST',
        header: {
            "Content-Type": "application/x-www-form-urlencoded"
        },
        success:res=>{
          setTimeout(function(){
            wx.hideLoading()
          }, 2000);
          console.log(res)
          console.log(res.data.container_all[100])
          this.setData({
            list:res.data.container_all
          })
          wx.setStorageSync('product_all', res.data.container_all)
        }
      })
    }
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
  //实时计算滚动高度
  scrollPosition(e) {
    const position = e.detail.scrollTop;
    this.setData({
      showBackTop: position > 500
    })
  },
  // 点击回到顶部
  onBackTop() {
    wx.pageScrollTo({
      scrollTop: 0,
      duration: 3000
    })
    this.setData({
      showBackTop: false,
      topPosition: 0,
    })
  },
  //开商品详情页
  showModal(e) {
    this.data.imgList.push('http://106.12.78.130:8080/pictures/PaddleClas/dataset/retail/'+e.currentTarget.dataset.product[3])
    this.setData({
      modalName: e.currentTarget.dataset.product,
      imgList:this.data.imgList,
      showBackTop: false
    })
    wx.setStorageSync('imglist', this.data.imgList)
  },
  //关商品详情页
  hideModal(e) {
    this.setData({
      modalName: null,
      imgList:[],
      isimageRevised:false,
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
          this.setData({
            imgList: res.tempFilePaths
          })
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
      imgList: this.data.imgList,
      isimageRevised:true
    })
  },
  //修改、删除按钮操作
  formSubmit: function (e){
    var that = this
    var formData = e.detail.value
    if(e.detail.target.dataset.choice=='revise'){
      //判断表单是否为空
      if(formData.productname&&formData.productunitprice&&this.data.imgList[0]!=null){
        wx.showModal({
          title: '修改确认',
          content: '请检查修改信息正确无误',
          cancelText: '取消',
          confirmText: '确认',
          success: res => {
            if (res.confirm) {
              //判断图片是否被修改
              if(this.data.isimageRevised){
                wx.showLoading({
                  title: '正在上传',
                })
                //上传表单
                wx.uploadFile({
                  url: 'http://106.12.78.130/replace/',
                  filePath: this.data.imgList[0],
                  name: 'productimage',
                  formData:{
                    'number':this.data.modalName[0],
                    'container_name':formData.productname,
                    'container_price':formData.productunitprice,
                    'isimageRevised':this.data.isimageRevised,
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
                    //判断是否修改成功
                    if(result.state=='true'){
                      app.globalData.sessionID = result.sessionID
                      wx.setStorageSync('sessionID', result.sessionID)
                      wx.showToast({
                        title: '上传成功',
                        icon:'success'
                      })
                    }
                    else{
                      wx.hideLoading()
                      wx.showToast({
                        title: '上传失败',
                        icon:'error'
                      })
                    }
                  }
                })
              }
              else{
                wx.showLoading({
                  title: '正在上传',
                })
                //提交表单
                wx.request({
                  url: 'http://106.12.78.130/replace/',
                  data:{
                    'number':this.data.modalName[0],
                    'container_name':formData.productname,
                    'container_price':formData.productunitprice,
                    'isimageRevised':this.data.isimageRevised,
                    'sessionID':wx.getStorageSync('sessionID'),
                    'isSKexpired':JSON.stringify(wx.getStorageSync('isSKexpired')),
                    'code':JSON.stringify(wx.getStorageSync('code'))
                  },
                  method: 'POST',
                  header: {
                      "Content-Type": "application/x-www-form-urlencoded"
                  },
                  success:res=>{
                    var result = res.data
                    console.log(result.state)
                    console.log(res)
                    wx.hideLoading()
                    if(result.state=='true'){
                      app.globalData.sessionID = result.sessionID
                      wx.setStorageSync('sessionID', result.sessionID)
                      wx.showToast({
                        title: '上传成功',
                        icon:'success'
                      })
                    }
                    else{
                      wx.hideLoading()
                      wx.showToast({
                        title: '上传失败',
                        icon:'error'
                      })
                    }
                  }
                })
              }
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
    }
    else if(e.detail.target.dataset.choice=='delete'){
      wx.showModal({
        title: '删除确认',
        content: '请确认删除信息操作',
        cancelText: '取消',
        confirmText: '确认',
        success: res => {
          if (res.confirm) {
            wx.showLoading({
              title: '正在删除',
            })
            console.log(this.data.modalName[0])
            wx.request({
              url: 'http://106.12.78.130/delete/',
              data:{
                'number':JSON.stringify(this.data.modalName[0]),
                'container_name':formData.productname,
                'container_price':formData.productunitprice,
                'sessionID':wx.getStorageSync('sessionID'),
                'isSKexpired':JSON.stringify(wx.getStorageSync('isSKexpired')),
                'code':JSON.stringify(wx.getStorageSync('code'))
              },
              method:'POST',
              header: {
                "Content-Type": "application/x-www-form-urlencoded"
              },
              success(res){
                //删除成功后执行
                var result = res.data
                wx.hideLoading()
                console.log(result)
                if(result.state=='true'){
                  app.globalData.sessionID = result.sessionID
                  wx.setStorageSync('sessionID', result.sessionID)
                  wx.showToast({
                    title: '删除成功',
                    icon:'success'
                  })
                  //修改缓存数据
                  var delete_index = that.data.modalName[0]
                  var product_all = wx.getStorageSync('product_all')
                  for(var i = 0;i < product_all.length;i++){
                    console.log(product_all[i][0])
                    if(product_all[i][0]==delete_index){
                      product_all.splice(i,1)
                    }
                  }
                  wx.setStorageSync('product_all', product_all)
                  that.setData({
                    list:product_all
                  })
                  //上传成功，清空表单
                  that.setData({
                    modalName:null,
                    imgList:[]
                  })
                }
                else{
                  wx.showToast({
                    title: '删除失败',
                    icon:'error'
                  })
                }
              }
            })
          }
        }
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