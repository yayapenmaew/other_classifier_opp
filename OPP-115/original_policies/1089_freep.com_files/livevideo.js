define(["jquery","underscore","backbone","pubsub","state","utils","modules/global/brightcove-video"],function(e,i,o,t,d,l,s){var n=o.View.extend({events:{"click #close_modal_link":"close"},initialize:function(i){this.m_uiModal={},this.m_uiModalBgrnd={},this.win=l.get("win")[0],this.body=l.get("body"),this.service=i&&i.service?i.service:"/services/livevideos/",this.win.gannett={};var o=this;this.win.gannett.videoplayer_modal={openPopup:function(){var i="/livevideos/popout/",t=e("#videoplayer_modal .video-title").text(),d=e("#videoplayer_modal .sponsor-text").text(),s=e("#videoplayer_modal .video-logo").attr("src"),n=e("#videoplayer_modal").attr("data-assetid");i+="?videoid="+n+"&description="+t+"&sponsor="+d+"&sourcelogo="+s,n&&(l.openPopup(i,640,440),o.close())}}},loadLiveVideo:function(i){var o=this;d.registerFullScreenView(this),d.fetchHtml(this.service+i+"/").done(function(i){o.body.append(i),o.video=new s({el:e("#videoplayer_modal"),autostart:!0,live:!0}),o.m_uiModal=e("#videoplayer_modal"),o.m_uiModalBgrnd=e("#lightbox"),o.m_uiModal.show(),o.m_uiModalBgrnd.fadeIn(300),o.setElement("#videoplayer_modal")})},close:function(){if(this.m_uiModal){d.clearFullScreenView();var e=this.m_uiModalBgrnd;this.destroy(!0),this.m_uiModalBgrnd.fadeOut(300,function(){e.remove()}),this.m_uiModal.remove(),this.m_uiModalBgrnd.remove(),this.m_uiModal=null,this.m_uiModalBgrnd=null,this.win.gannett={}}},destroy:function(e){this.video&&this.video.destroy(!0),this.undelegateEvents(),e&&this.remove()}});return n});
//# sourceMappingURL=livevideo.js
//# sourceMappingURL=livevideo.js.map