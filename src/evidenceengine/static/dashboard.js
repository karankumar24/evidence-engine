/* EvidenceEngine dashboard — Alpine.js component definitions */

document.addEventListener('alpine:init', () => {
    Alpine.data('reviewKeyboard', () => ({
        get activeClaim() {
            return document.querySelector('.claim-row.bg-slate-100');
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
            const active = document.querySelector('.claim-row.bg-slate-100');
            const idx = rows.indexOf(active);
            if (idx >= 0 && idx < rows.length - 1) rows[idx + 1].click();
            else if (rows.length > 0 && idx === -1) rows[0].click();
        },
        prevClaim() {
            const rows = Array.from(document.querySelectorAll('.claim-row'));
            const active = document.querySelector('.claim-row.bg-slate-100');
            const idx = rows.indexOf(active);
            if (idx > 0) rows[idx - 1].click();
        },
    }));
});
