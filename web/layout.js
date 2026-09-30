/**
 * layout.js — injects shared nav, footer, and font preloads into every page.
 * Called via <script src="/web/layout.js"></script> in <head> of every HTML page.
 */
(function () {
  // Inject Google Fonts preconnect + stylesheet into <head>
  function injectFonts() {
    const preconnect1 = document.createElement('link');
    preconnect1.rel = 'preconnect';
    preconnect1.href = 'https://fonts.googleapis.com';
    document.head.appendChild(preconnect1);

    const preconnect2 = document.createElement('link');
    preconnect2.rel = 'preconnect';
    preconnect2.href = 'https://fonts.gstatic.com';
    preconnect2.crossOrigin = 'anonymous';
    document.head.appendChild(preconnect2);

    const fontLink = document.createElement('link');
    fontLink.rel = 'stylesheet';
    fontLink.href =
      'https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,600;0,700;1,400&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap';
    document.head.appendChild(fontLink);
  }

  // Determine active page from path
  function getActivePage() {
    const path = window.location.pathname;
    if (path === '/' || path === '/index.html') return 'dashboard';
    if (path.includes('performance')) return 'performance';
    if (path.includes('about')) return 'about';
    return '';
  }

  function isActive(page, current) {
    return page === current ? 'aria-current="page"' : '';
  }

  function renderNav() {
    const current = getActivePage();

    const nav = document.createElement('nav');
    nav.className = 'nav-bar';
    nav.setAttribute('aria-label', 'Main navigation');
    nav.innerHTML = `
      <div class="nav-inner">
        <a href="/" class="nav-brand" aria-label="ClimateGuard home">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M12 2a7 7 0 0 1 7 7c0 3.5-2.5 6.5-5.5 8.5a3 3 0 0 1-3 0C7.5 15.5 5 12.5 5 9a7 7 0 0 1 7-7z"/>
            <circle cx="12" cy="9" r="2.5"/>
          </svg>
          ClimateGuard
        </a>
        <div class="nav-links" role="list">
          <a href="/"            class="nav-link" ${isActive('dashboard',   current)} role="listitem">Dashboard</a>
          <a href="/performance" class="nav-link" ${isActive('performance', current)} role="listitem">Model performance</a>
          <a href="/about"       class="nav-link" ${isActive('about',       current)} role="listitem">About</a>
        </div>
      </div>
    `;
    return nav;
  }

  function renderFooter() {
    const footer = document.createElement('footer');
    footer.className = 'site-footer';
    footer.innerHTML = `
      <div class="site-footer__inner">
        <p class="site-footer__text">
          ClimateGuard &mdash; Heatwave risk forecasts for five Indian cities.
          Built with ERA5 reanalysis data (1990&ndash;2025).
        </p>
        <p class="site-footer__disclaimer">
          <strong>Research project only.</strong> This model uses an IMD-inspired heatwave definition
          applied to ERA5 reanalysis data, not official IMD station observations. Probability
          estimates and risk levels are project-defined and are not official government warnings.
          During heat emergencies, follow advisories from the
          <a href="https://mausam.imd.gov.in/" target="_blank" rel="noopener noreferrer">India Meteorological Department</a>.
        </p>
      </div>
    `;
    return footer;
  }

  function renderSkipLink() {
    const a = document.createElement('a');
    a.href = '#main-content';
    a.className = 'skip-link';
    a.textContent = 'Skip to main content';
    return a;
  }

  // Wait for DOM then inject
  function inject() {
    injectFonts();

    const body = document.body;

    // Skip link goes first
    body.prepend(renderSkipLink());

    // Nav after skip link
    body.insertBefore(renderNav(), body.firstChild.nextSibling);

    // Footer at end
    body.appendChild(renderFooter());

    // Remove any hard-coded nav-bar / site-footer already in HTML
    // (during transition period some pages may still have them)
    document.querySelectorAll('.nav-bar-static, .footer').forEach(el => el.remove());
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', inject);
  } else {
    inject();
  }
})();
