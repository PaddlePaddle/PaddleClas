//app.js
// import paddlejs, { runner } from 'paddlejs';
// var plugin = requirePlugin("paddlejs-plugin");
App({
    onLaunch: function() {
        if (!wx.cloud) {
            console.error('请使用 2.2.3 或以上的基础库以使用云能力')
        } else {
            wx.cloud.init({
                env: 'garbagesort-7gtytassc65e553c',
                traceUser: true,
            })
        }
        const updateManager = wx.getUpdateManager()
        updateManager.onCheckForUpdate(function(res) {
            console.log(res.hasUpdate)
            if (res.hasUpdate) {
                updateManager.onUpdateReady(function() {
                    wx.showModal({
                        title: '更新提示',
                        content: '新版本已经准备好，是否重启应用？',
                        success: function(res) {
                            if (res.confirm) {
                                updateManager.applyUpdate()
                            }
                        }
                    })
                })
            }
        })
        updateManager.onUpdateFailed(function() {
            // 新版本下载失败
        })
        // colorui获取系统信息
        wx.getSystemInfo({
            success: e => {
              this.globalData.StatusBar = e.statusBarHeight;
              let custom = wx.getMenuButtonBoundingClientRect();
              this.globalData.Custom = custom;  
              this.globalData.CustomBar = custom.bottom + custom.top - e.statusBarHeight;
              this.globalData.statusBarHeight = e.statusBarHeight;
            }
        })
        // wx.checkSession({
        //   success(){
        //       this.globalData.isSKexpired = false
        //       this.globalData.code = null
        //   },
        //   fail(){
        //       this.globalData.isSKexpired = true
        //   }
        // })
        // plugin.register(paddlejs,wx);
    },
    
    
    globalData: {
        userInfo:null,
        sessionid:null,
        code:null,
        isSKexpired:null,
        statusBarHeight: 0
        // Paddlejs: paddlejs.runner
    },
})