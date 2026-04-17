/* EvidenceEngine dashboard — Alpine.js component definitions */

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

// Exposed as a global so base.html can call reviewKeyboard() in x-data expression.
// Also registered with Alpine.data for component-name syntax.
window.reviewKeyboard = () => ({
    helpOpen: false,
    get activeClaim() {
        return document.querySelector('.claim-row[data-active]');
    },
    approve() {
        const btn = document.querySelector('#detail-panel .btn-approve');
        if (btn) btn.click();
    },
    reject() {
        const btn = document.querySelector('#detail-panel .btn-reject');
        if (btn) btn.click();
    },
    flag() {
        const btn = document.querySelector('#detail-panel .btn-mark_insufficient');
        if (btn) btn.click();
    },
    nextClaim() {
        const rows = Array.from(document.querySelectorAll('.claim-row'));
        const active = this.activeClaim;
        const idx = rows.indexOf(active);
        if (idx >= 0 && idx < rows.length - 1) rows[idx + 1].click();
        else if (rows.length > 0 && idx === -1) rows[0].click();
    },
    prevClaim() {
        const rows = Array.from(document.querySelectorAll('.claim-row'));
        const active = this.activeClaim;
        const idx = rows.indexOf(active);
        if (idx > 0) rows[idx - 1].click();
    },
});

document.addEventListener('alpine:init', () => {
    Alpine.data('reviewKeyboard', window.reviewKeyboard);
});
