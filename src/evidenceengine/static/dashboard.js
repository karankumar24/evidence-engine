/* EvidenceEngine dashboard — event handlers for the reviewer UI */

/* Track active queue row via data-active attribute, set on HTMX trigger. */
document.addEventListener('htmx:beforeRequest', (e) => {
    const row = e.target.closest('.claim-row');
    if (!row) return;
    document.querySelectorAll('.claim-row[data-active]').forEach(r => {
        r.removeAttribute('data-active');
        r.classList.remove('bg-paper-200');
        r.classList.add('hover:bg-paper-100');
    });
    row.setAttribute('data-active', 'true');
    row.classList.add('bg-paper-200');
    row.classList.remove('hover:bg-paper-100');
});

/* Show a dismissable toast when HTMX gets a non-2xx response or network error. */
function _showToast(message) {
    const host = document.getElementById('htmx-toast');
    if (!host) return;
    const el = document.createElement('div');
    el.setAttribute('role', 'alert');
    el.className = 'mb-2 px-4 py-3 bg-verdict-contradicted text-paper-50 border border-verdict-contradicted rounded-sm text-sm font-sans shadow-md flex items-start gap-3';
    el.innerHTML = `<span class="flex-1">${message}</span>
                    <button type="button" aria-label="Dismiss" class="text-paper-50/80 hover:text-paper-50 text-base leading-none">×</button>`;
    el.querySelector('button').addEventListener('click', () => el.remove());
    host.appendChild(el);
    setTimeout(() => el.remove(), 6000);
}

document.addEventListener('htmx:responseError', (e) => {
    const status = e.detail && e.detail.xhr ? e.detail.xhr.status : '?';
    _showToast(`Request failed (HTTP ${status}). Please retry.`);
});

document.addEventListener('htmx:sendError', () => {
    _showToast('Network error. Check your connection and retry.');
});

/* Mark queue rows that have been reviewed so CSS can dim them. */
document.addEventListener('htmx:afterSwap', () => {
    document.querySelectorAll('.claim-row').forEach(row => {
        const status = row.querySelector('[id$="-status"]');
        if (!status) return;
        const label = status.textContent.trim().toLowerCase();
        if (label && label !== 'unreviewed') {
            row.setAttribute('data-reviewed', 'true');
        } else {
            row.removeAttribute('data-reviewed');
        }
    });
});

/* Keyboard shortcuts: A/R/F/J/K — native listeners, no Alpine dependency. */
function _activeRow() {
    return document.querySelector('.claim-row[data-active]');
}

document.addEventListener('keydown', (e) => {
    // Skip when focus is in an input, textarea, or select
    const tag = document.activeElement && document.activeElement.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    // Skip when modifier keys are held (cmd/ctrl shortcuts)
    if (e.metaKey || e.ctrlKey) return;

    if (e.key === 'a') {
        e.preventDefault();
        const btn = document.querySelector('#detail-panel .btn-approve');
        if (btn) btn.click();
    } else if (e.key === 'r') {
        e.preventDefault();
        const btn = document.querySelector('#detail-panel .btn-reject');
        if (btn) btn.click();
    } else if (e.key === 'f') {
        e.preventDefault();
        const btn = document.querySelector('#detail-panel .btn-mark_insufficient');
        if (btn) btn.click();
    } else if (e.key === 'j') {
        e.preventDefault();
        const rows = Array.from(document.querySelectorAll('.claim-row'));
        const active = _activeRow();
        const idx = rows.indexOf(active);
        if (idx >= 0 && idx < rows.length - 1) rows[idx + 1].click();
        else if (rows.length > 0 && idx === -1) rows[0].click();
    } else if (e.key === 'k') {
        e.preventDefault();
        const rows = Array.from(document.querySelectorAll('.claim-row'));
        const active = _activeRow();
        const idx = rows.indexOf(active);
        if (idx > 0) rows[idx - 1].click();
    }
});
