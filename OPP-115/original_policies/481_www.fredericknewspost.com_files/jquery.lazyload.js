;(function(a,b){var c=a(b);a.fn.lazyload=function(d){function h(){var b=0;e.each(function(){var c=a(this);if(g.skip_invisible&&!c.is(":visible")){return}if(a.abovethetop(this,g)||a.leftofbegin(this,g)){}else if(!a.belowthefold(this,g)&&!a.rightoffold(this,g)){c.trigger("appear")}else{if(++b>g.failure_limit){return false}}})}var e=this;var f;var g={threshold:0,failure_limit:0,event:"scroll",effect:"show",container:b,data_attribute:"original",skip_invisible:true,appear:null,load:null};if(d){if(undefined!==d.failurelimit){d.failure_limit=d.failurelimit;delete d.failurelimit}if(undefined!==d.effectspeed){d.effect_speed=d.effectspeed;delete d.effectspeed}a.extend(g,d)}f=g.container===undefined||g.container===b?c:a(g.container);if(0===g.event.indexOf("scroll")){f.bind(g.event,function(a){return h()})}this.each(function(){var b=this;var c=a(b);b.loaded=false;c.one("appear",function(){if(!this.loaded){if(g.appear){var d=e.length;g.appear.call(b,d,g)}a("<img />").bind("load",function(){c.hide().attr("src",c.data(g.data_attribute))[g.effect](g.effect_speed);b.loaded=true;var d=a.grep(e,function(a){return!a.loaded});e=a(d);if(g.load){var f=e.length;g.load.call(b,f,g)}}).attr("src",c.data(g.data_attribute))}});if(0!==g.event.indexOf("scroll")){c.bind(g.event,function(a){if(!b.loaded){c.trigger("appear")}})}});c.bind("resize",function(a){h()});h();return this};a.belowthefold=function(d,e){var f;if(e.container===undefined||e.container===b){f=c.height()+c.scrollTop()}else{f=a(e.container).offset().top+a(e.container).height()}return f<=a(d).offset().top-e.threshold};a.rightoffold=function(d,e){var f;if(e.container===undefined||e.container===b){f=c.width()+c.scrollLeft()}else{f=a(e.container).offset().left+a(e.container).width()}return f<=a(d).offset().left-e.threshold};a.abovethetop=function(d,e){var f;if(e.container===undefined||e.container===b){f=c.scrollTop()}else{f=a(e.container).offset().top}return f>=a(d).offset().top+e.threshold+a(d).height()};a.leftofbegin=function(d,e){var f;if(e.container===undefined||e.container===b){f=c.scrollLeft()}else{f=a(e.container).offset().left}return f>=a(d).offset().left+e.threshold+a(d).width()};a.inviewport=function(b,c){return!a.rightofscreen(b,c)&&!a.leftofscreen(b,c)&&!a.belowthefold(b,c)&&!a.abovethetop(b,c)};a.extend(a.expr[":"],{"below-the-fold":function(b){return a.belowthefold(b,{threshold:0})},"above-the-top":function(b){return!a.belowthefold(b,{threshold:0})},"right-of-screen":function(b){return a.rightoffold(b,{threshold:0})},"left-of-screen":function(b){return!a.rightoffold(b,{threshold:0})},"in-viewport":function(b){return!a.inviewport(b,{threshold:0})},"above-the-fold":function(b){return!a.belowthefold(b,{threshold:0})},"right-of-fold":function(b){return a.rightoffold(b,{threshold:0})},"left-of-fold":function(b){return!a.rightoffold(b,{threshold:0})}})})(jQuery,window);