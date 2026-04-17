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

document.addEventListener('alpine:init', () => {
    Alpine.data('reviewKeyboard', () => ({
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
    }));
});
