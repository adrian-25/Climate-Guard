/* Lightweight, accessible double-opt-in form. Existing dashboard controls stay untouched. */
(() => {
  const form = document.querySelector('#subscribe-form');
  if (!form) return;
  const city = document.querySelector('#subscribe-city');
  const status = document.querySelector('#subscribe-status');
  fetch('/api/cities').then((r) => r.json()).then((cities) => {
    city.innerHTML = cities.map((item) => `<option value="${item.key}">${item.name}</option>`).join('');
  }).catch(() => { status.textContent = 'Cities are unavailable right now. Please try again later.'; });
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!form.reportValidity()) return;
    status.textContent = 'Sending confirmation…';
    const payload = Object.fromEntries(new FormData(form));
    try {
      const response = await fetch('/api/subscribe', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || 'Unable to subscribe.');
      status.textContent = result.message;
      form.reset();
    } catch (error) { status.textContent = error.message; }
  });
})();
