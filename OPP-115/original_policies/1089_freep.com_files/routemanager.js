define(["jquery","underscore","state","managers/siteconfig","pubsub","utils"],function(t,n,e,i,a,o){"use strict";var r=function(){};return r.prototype={baseUrl:window.location.protocol+"//"+window.location.host,usatBetaDomain:"http://beta.usatoday.com",start:function(t){this.started=!0,i.loadRoutes().done(n.bind(function(){this.started&&(t&&o.get("win").on("popstate",n.bind(this._onStateChange,this)),this._onStateChange(),console.log("Site Loaded"))},this))},getLocalPath:function(t){if(!t||"#"===t)return null;if(t.indexOf(this.baseUrl))if(t.indexOf(this.usatBetaDomain)){if(-1!==t.indexOf("://"))return null}else t=t.substring(this.usatBetaDomain.length);else t=t.substring(this.baseUrl.length);return t},stop:function(){this.started=!1,o.get("win").off("popstate")},getInfo:function(n){var e=this.getLocalPath(n);return null!==e?i.getRouteInfo(t.trim(e)):null},isAjax:function(t){var n=this.getInfo(t);return n&&n.ajax},goTo:function(t,n,i){if(!o.isValidUrl(t)||!this._canNavigate())return!1;if(!e.getIntentUrl()&&this.isCurrentUrl(t))return console.warn("Tried navigating to the current url, skipping"),!1;a.trigger("track",{label:n,url:t,event:i});var r=this._goHashTag(t)||this._goInternal(t);return r||this._setLocation(this.attachChromeless(t)),!0},goToInterstitual:function(t){return t=this.getLocalPath(t),Modernizr.history&&o.isValidUrl(t)&&this._canNavigate()&&this.getInfo(t)?(e.setIntentUrl(t),this._pushState(this.attachChromeless(t)),!0):!1},isCurrentUrl:function(t){var n=this.getLocalPath(t);return n&&(n=n.split("#")[0]),n===this._getCurrentPath()},_getCurrentPath:function(){return window.location.pathname+location.search},_setLocation:function(t){window.location.assign(t)},_canNavigate:function(){var t=e.getActivePageInfo().navigationWarning;return t&&!window.confirm(t)?!1:!0},_goHashTag:function(n){if("#"===n[0]){n=n.substring(1);var e=t("a[name="+n+"]").offset();return e&&e.top&&a.trigger("scrollTop",e.top-40),window.location.hash=n,!0}},attachChromeless:function(t){return window.chromeless?t+(-1===t.indexOf("?")?"?":"&")+"chromeless=true":t},_goInternal:function(t){return Modernizr.history&&this.isAjax(t)&&this.isAjax(o.getPageUrl())?(this._pushState(this.attachChromeless(this.getLocalPath(t))),this._onStateChange(),!0):!1},_pushState:function(t){window.history.pushState({},document.title,t)},_onStateChange:function(){var t,n=window.location;n.pathname!==this._currentPathName&&(this._currentPathName=n.pathname+n.search,t=this.getInfo(n.pathname),t?e.onRouteChange(t,n.pathname+n.search+n.hash):console.error("Could not find any site-config matching "+n.pathname))}},new r});
//# sourceMappingURL=routemanager.js
//# sourceMappingURL=routemanager.js.map