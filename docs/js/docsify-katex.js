(($docsify,katex) => {
	if(!$docsify){console.error("$docsify no exist.")}
	if(!katex){console.error("$katex no exist.")}
	if ($docsify && katex) {
		$docsify.plugins = [].concat(docsifyKatex(), $docsify.plugins)
	}
  
	function docsifyKatex() {
		return function(hook, vm) {
  
			hook.beforeEach(function(content) {

				const inline = content.replace(/\$\$\$([\s\S]*?)\$\$\$/g, function(m, code) {

					try {
						return `<katex><p class="ownline-math">${katex.renderToString(code.trim())}</p></katex>`;
					} catch (error) {
						return `<span style=\"color: red; font-weight: 500;\">${error.toString()}</span>`
					}

				});

				return inline.replace(/\$\$([\s\S]*?)\$\$/g, function(m, code) {

					try {
						return `<katex>${katex.renderToString(code.trim())}</katex>`;
					} catch (error) {
						return `<span style=\"color: red; font-weight: 500;\">${error.toString()}</span>`
					}

				});
				
			})
	
			hook.afterEach(function(html, next) {
				let parsed = html.replace(/<katex>([\s\S]*?)<\/katex>/g, function(m, code) {
					return code;
				});
				next(parsed);
			})
	
		};
	}
  
})(window.$docsify,window.katex);