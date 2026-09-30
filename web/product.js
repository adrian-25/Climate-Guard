/* PWA/offline, URL state, language choice, CSV export, and print helpers. */
(() => {
  const params = new URLSearchParams(location.search);
  const language = params.get('lang') || localStorage.getItem('climateguard-language') || navigator.language.slice(0, 2);
  const chosen = ['en', 'hi', 'mr'].includes(language) ? language : 'en';
  async function init() {
    const labels = await fetch('/web/locales.json').then(r => r.json());
    const text = labels[chosen];
    document.documentElement.lang = chosen;
    document.querySelectorAll('[data-i18n]').forEach(node => { if (text[node.dataset.i18n]) node.textContent = text[node.dataset.i18n]; });
    const tools = document.createElement('div'); tools.className = 'product-tools';
    tools.innerHTML = `<label>${text.language}<select id="language-picker"><option value="en">English</option><option value="hi">हिन्दी*</option><option value="mr">मराठी*</option></select></label><button id="export-city" type="button">${text.export}</button><button id="print-report" type="button">${text.print}</button>`;
    document.querySelector('.masthead')?.append(tools);
    document.querySelector('#language-picker').value = chosen;
    document.querySelector('#language-picker').addEventListener('change', e => { localStorage.setItem('climateguard-language', e.target.value); params.set('lang', e.target.value); location.search = params; });
    document.querySelector('#print-report').addEventListener('click', () => print());
    document.querySelector('#export-city').addEventListener('click', async () => {
      const city = window.ClimateGuardState?.selectedCity || params.get('city'); if (!city) return;
      const data = await fetch(`/api/trends/${encodeURIComponent(city)}`).then(r => r.json());
      const rows = [['date', 'tmax', 'tmin', 'probability', 'heatwave'], ...data.data.map(d => [d.date, d.tmax, d.tmin, d.prob, d.heatwave])];
      const blob = new Blob([rows.map(row => row.join(',')).join('\n')], { type: 'text/csv' });
      const link = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: `climateguard-${city}.csv` }); link.click(); URL.revokeObjectURL(link.href);
    });
    if (!navigator.onLine) document.body.insertAdjacentHTML('afterbegin', `<p class="offline-banner" role="status">${text.offline}</p>`);
  }
  if ('serviceWorker' in navigator) addEventListener('load', () => navigator.serviceWorker.register('/web/service-worker.js'));
  init().catch(() => {});
})();
